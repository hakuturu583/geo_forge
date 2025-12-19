from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import math

import gsplat
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from gsplat.strategy import DefaultStrategy
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils.gaussians import Gaussians3D, apply_transform, save_ply

from geo_forge.dataset import GeoForgeDataset, NuScenesData
from geo_forge.preprocess.sharp_preprocessor import predict_image

DEFAULT_SHARP_CHECKPOINT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)
_NUSC_CAM_TO_OPENGL = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)
_NUSC_WORLD_TO_GS = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


@dataclass
class LossWeightConfig:
    sky: float = 0.0
    movable_objects: float = 0.1


@dataclass
class DefaultStrategyConfig:
    prune_opacity_threshold: float = 0.001
    grow_grad2d_threshold: float = 5e-5
    refine_start_iter: int = 250
    refine_stop_iter: int = 15000
    reset_every: int = 2000


@dataclass
class SequenceTrainConfig:
    scene: str
    camera: str = "cam_front"
    nuscenes_version: str = os.getenv("NUSCENES_VERSION", "v1.0-mini")
    start_sample_index: int = 0
    start_timestamp: int | None = None
    steps: int = 0
    lr: float = 5e-3
    device: str = "default"
    log_interval: int = 10
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)
    strategy: DefaultStrategyConfig = field(default_factory=DefaultStrategyConfig)
    sharp_checkpoint: Path | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_render_interval: int = 1000
    wandb_render_frames: int = 16


def _select_device(device_pref: str) -> torch.device:
    if device_pref in ("", "default", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_pref)


def _load_sharp_predictor(
    checkpoint_path: Path | None, device: torch.device
) -> RGBGaussianPredictor:
    if checkpoint_path is None:
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_SHARP_CHECKPOINT_URL, progress=True
        )
    else:
        state_dict = torch.load(checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def _prepare_image_tensor(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Expected image tensor shape (3, H, W); got {tuple(image.shape)}"
        )
    return (
        image.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _gaussians_to_world(gaussians: Gaussians3D, c2w_gs: torch.Tensor) -> Gaussians3D:
    if c2w_gs.shape != (4, 4):
        raise ValueError("c2w must be shape (4, 4).")
    device = gaussians.mean_vectors.device
    c2w_gs = c2w_gs.to(device=device, dtype=torch.float32)
    inv_world_to_gs = _NUSC_WORLD_TO_GS.to(device=device)
    inv_cam_to_opengl = _NUSC_CAM_TO_OPENGL.to(device=device)
    c2w_nusc = inv_world_to_gs @ c2w_gs @ inv_cam_to_opengl
    return apply_transform(gaussians, c2w_nusc[:3, :])


def _flatten_gaussians(gaussians: Gaussians3D) -> Gaussians3D:
    mean_vectors = gaussians.mean_vectors
    if mean_vectors.dim() == 3:
        mean_vectors = mean_vectors.flatten(0, 1)

    singular_values = gaussians.singular_values
    if singular_values.dim() == 3:
        singular_values = singular_values.flatten(0, 1)

    quaternions = gaussians.quaternions
    if quaternions.dim() == 3:
        quaternions = quaternions.flatten(0, 1)

    colors = gaussians.colors
    if colors.dim() == 3:
        colors = colors.flatten(0, 1)

    opacities = gaussians.opacities
    if opacities.dim() == 2:
        opacities = opacities.flatten(0, 1)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def _initialize_params_from_gaussians(
    gaussians: Gaussians3D, device: torch.device
) -> dict[str, torch.nn.Parameter]:
    gaussians = _flatten_gaussians(gaussians)

    positions = gaussians.mean_vectors.detach().to(device=device, dtype=torch.float32)
    scales_log = (
        gaussians.singular_values.detach()
        .to(device=device, dtype=torch.float32)
        .clamp_min(1e-12)
        .log()
    )
    quats = gaussians.quaternions.detach().to(device=device, dtype=torch.float32)
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)
    colors = gaussians.colors.detach().to(device=device, dtype=torch.float32)
    opacities_raw = gaussians.opacities.detach().to(device=device, dtype=torch.float32)
    if opacities_raw.dim() == 2 and opacities_raw.shape[-1] == 1:
        opacities_raw = opacities_raw.squeeze(-1)
    opacities = torch.logit(opacities_raw.clamp(1e-6, 1.0 - 1e-6))

    return {
        "means": torch.nn.Parameter(positions),
        "scales": torch.nn.Parameter(scales_log),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }


def _params_to_gaussians(params: dict[str, torch.nn.Parameter]) -> Gaussians3D:
    means = params["means"].detach().unsqueeze(0)
    scales = torch.exp(params["scales"].detach()).unsqueeze(0)
    quats = params["quats"].detach()
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)
    quats = quats.unsqueeze(0)
    colors = params["colors"].detach().unsqueeze(0)
    opacities = torch.sigmoid(params["opacities"].detach()).unsqueeze(0)
    return Gaussians3D(
        mean_vectors=means,
        singular_values=scales,
        quaternions=quats,
        colors=colors,
        opacities=opacities,
    )


def _save_window_gaussians(
    params: dict[str, torch.nn.Parameter],
    sample: NuScenesData,
    *,
    window_step: str,
) -> None:
    dataset_root = os.getenv("GEOFORGE_DATASET_ROOT")
    if not dataset_root:
        raise EnvironmentError(
            "GEOFORGE_DATASET_ROOT must be set to save merged Gaussians."
        )
    scene = str(sample["scene"])
    camera = str(sample["camera"])
    output_dir = Path(dataset_root) / scene / camera / "merged_gaussians"
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = sample["intrinsics"]
    f_px = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    width = int(sample["width"])
    height = int(sample["height"])

    ply_path = output_dir / f"{window_step}.ply"
    gaussians = _params_to_gaussians(params)
    save_ply(gaussians, f_px, (height, width), ply_path)
    print(f"[save {window_step}] saved {ply_path}")


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (image * 255.0).round().astype(np.uint8)


def _log_wandb_media(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    step: int,
    render_buffer: list[np.ndarray],
    config: SequenceTrainConfig,
) -> None:
    render_buffer.append(_to_uint8_image(pred))
    if len(render_buffer) > int(config.wandb_render_frames):
        render_buffer.pop(0)

    render_interval = int(config.wandb_render_interval)
    if render_interval <= 0 or step % render_interval != 0:
        return

    from PIL import Image
    import tempfile

    if not render_buffer:
        return

    gif_frames = [Image.fromarray(frame) for frame in render_buffer]
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_frames[0].save(
            tmp.name,
            save_all=True,
            append_images=gif_frames[1:],
            duration=150,
            loop=0,
        )
        wandb.log(
            {
                "render/gif": wandb.Video(tmp.name, format="gif"),
                "render/target": wandb.Image(_to_uint8_image(target)),
            },
            step=step,
        )


def _build_loss_weights(
    sample: NuScenesData,
    config: SequenceTrainConfig,
    *,
    device: torch.device,
    height: int,
    width: int,
) -> torch.Tensor:
    sky_mask = sample.get("sky_mask")
    object_mask = sample.get("object_mask")

    loss_weights = torch.ones((1, height, width), device=device)
    if sky_mask is not None:
        loss_weights = torch.where(
            sky_mask.to(device).unsqueeze(0).bool(),
            torch.tensor(config.loss_weights.sky, device=device),
            loss_weights,
        )
    else:
        raise RuntimeError(
            "sky_mask is required but not provided in the sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )

    if object_mask is not None:
        obj_mask = object_mask.to(device).unsqueeze(0)
        loss_weights = torch.where(
            obj_mask.bool(),
            torch.tensor(config.loss_weights.movable_objects, device=device),
            loss_weights,
        )
    else:
        raise RuntimeError(
            "object_mask is required but not provided in the sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )
    return loss_weights


def _masked_l1_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    sample: NuScenesData,
    config: SequenceTrainConfig,
    device: torch.device,
) -> torch.Tensor:
    _, _, height, width = pred.unsqueeze(0).shape
    loss_weights = _build_loss_weights(
        sample, config, device=device, height=height, width=width
    )
    loss_map = F.l1_loss(pred, target, reduction="none")
    weights = loss_weights.expand_as(loss_map).to(loss_map.dtype)
    weight_sum = weights.sum()
    if weight_sum.item() == 0:
        return loss_map.new_tensor(0.0)
    return (loss_map * weights).sum() / weight_sum


def _render_gaussians(
    params: dict[str, torch.nn.Parameter],
    sample: NuScenesData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | int]]:
    image = sample["image"].to(device)
    intrinsics = sample["intrinsics"].to(device)
    c2w = sample["c2w"].to(device)
    width = int(sample["width"])
    height = int(sample["height"])

    scales = torch.exp(params["scales"])
    opacities = torch.sigmoid(params["opacities"])

    viewmat = torch.inverse(c2w)[None, ...]
    Ks = intrinsics[None, ...]
    (radii, means2d, depths, conics, _) = gsplat.rendering.fully_fused_projection(
        means=params["means"],
        covars=None,
        quats=params["quats"],
        scales=scales,
        viewmats=viewmat,
        Ks=Ks,
        width=width,
        height=height,
        opacities=opacities,
    )

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = gsplat.rendering.isect_tiles(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        sort=True,
        segmented=False,
        packed=False,
    )
    isect_offsets = gsplat.rendering.isect_offset_encode(
        isect_ids=isect_ids,
        n_images=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    backgrounds = torch.zeros(1, 3, device=device)
    pred, _ = gsplat.rendering.rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=params["colors"][None, ...],
        opacities=opacities[None, ...],
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        packed=False,
        absgrad=False,
    )
    if pred.dim() == 5:
        pred = pred.permute(0, 1, 4, 2, 3)[0, 0]
    elif pred.dim() == 4:
        pred = pred.permute(0, 3, 1, 2)[0]
    info = {
        "means2d": means2d,
        "width": width,
        "height": height,
        "n_cameras": 1,
        "radii": radii,
        "gaussian_ids": torch.arange(params["means"].shape[0], device=device).unsqueeze(
            0
        ),
    }
    return pred, image, info


def _select_start_sample(
    dataset: GeoForgeDataset, config: SequenceTrainConfig
) -> NuScenesData:
    sample_metas = sorted(
        list(enumerate(dataset.samples)),
        key=lambda item: int(item[1]["timestamp"]),
    )
    if not sample_metas:
        raise RuntimeError("No sample frames available in the dataset.")
    if config.start_timestamp is not None:
        for idx, meta in sample_metas:
            if int(meta["timestamp"]) >= int(config.start_timestamp):
                return dataset[idx]
        raise ValueError("start_timestamp is after the last sample frame.")

    idx = int(config.start_sample_index)
    if idx < 0 or idx >= len(sample_metas):
        raise ValueError(
            f"start_sample_index out of range: {idx} (0..{len(sample_metas)-1})"
        )
    return dataset[sample_metas[idx][0]]


def train_sequence(config: SequenceTrainConfig) -> None:
    scene_filter = [config.scene]
    camera_filter = [config.camera.lower()]

    sample_dataset = GeoForgeDataset(
        version=config.nuscenes_version,
        scene_filter=scene_filter,
        camera_filter=camera_filter,
        only_sample_frames=True,
    )
    sweep_dataset = GeoForgeDataset(
        version=config.nuscenes_version,
        scene_filter=scene_filter,
        camera_filter=camera_filter,
        only_sample_frames=False,
    )

    device = _select_device(config.device)
    start_sample = _select_start_sample(sample_dataset, config)
    start_ts = int(start_sample["timestamp"])

    predictor = _load_sharp_predictor(config.sharp_checkpoint, device=device)
    image_np = _prepare_image_tensor(start_sample["image"])
    with torch.no_grad():
        init_gaussians = predict_image(
            predictor, image_np, intrinsics=start_sample["intrinsics"], device=device
        )
    init_gaussians = _gaussians_to_world(init_gaussians, start_sample["c2w"])
    params = _initialize_params_from_gaussians(init_gaussians, device=device)

    base_lr = float(config.lr)
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=base_lr * 0.032, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=base_lr * 1.0, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=base_lr * 0.2, eps=1e-15),
        "opacities": torch.optim.Adam(
            [params["opacities"]], lr=base_lr * 10.0, eps=1e-15
        ),
        "colors": torch.optim.Adam([params["colors"]], lr=base_lr * 0.5, eps=1e-15),
    }
    strategy = DefaultStrategy(
        verbose=True,
        prune_opa=config.strategy.prune_opacity_threshold,
        grow_grad2d=config.strategy.grow_grad2d_threshold,
        refine_start_iter=config.strategy.refine_start_iter,
        refine_stop_iter=config.strategy.refine_stop_iter,
        reset_every=config.strategy.reset_every,
    )
    strategy_state = strategy.initialize_state()
    strategy.check_sanity(params, optimizers)

    use_wandb = bool(config.wandb_project)
    render_buffer: list[np.ndarray] = []
    if use_wandb:
        run_name = config.wandb_run_name or datetime.now().isoformat(timespec="seconds")
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "scene": config.scene,
                "camera": config.camera,
                "steps": config.steps,
                "lr": config.lr,
                "render_interval": config.wandb_render_interval,
                "render_frames": config.wandb_render_frames,
                "loss_log_interval": 10,
                "strategy": {
                    "prune_opa": config.strategy.prune_opacity_threshold,
                    "grow_grad2d": config.strategy.grow_grad2d_threshold,
                    "refine_start": config.strategy.refine_start_iter,
                    "refine_stop": config.strategy.refine_stop_iter,
                    "reset_every": config.strategy.reset_every,
                },
            },
        )

    sweep_indices = [
        idx
        for idx, meta in enumerate(sweep_dataset.samples)
        if int(meta["timestamp"]) > start_ts
    ]
    if not sweep_indices:
        print("No sweep frames found after the start sample frame.")
        return

    sweep_indices.sort(key=lambda idx: int(sweep_dataset.samples[idx]["timestamp"]))
    steps = int(config.steps) if int(config.steps) > 0 else len(sweep_indices)
    print(f"[full sequence] samples={len(sweep_indices)} steps={steps}")

    global_step = 0
    for step in range(steps):
        for opt in optimizers.values():
            opt.zero_grad()

        sample = sweep_dataset[sweep_indices[step % len(sweep_indices)]]
        pred, target, info = _render_gaussians(params, sample, device=device)
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=global_step,
            info=info,
        )
        loss = _masked_l1_loss(
            pred=pred,
            target=target,
            sample=sample,
            config=config,
            device=device,
        )
        loss.backward()

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=global_step,
            info=info,
        )

        if params["means"].shape[0] == 0:
            raise RuntimeError(
                "All Gaussians were pruned. Relax pruning thresholds or "
                "initialize with more Gaussians."
            )

        for opt in optimizers.values():
            opt.step()
        with torch.no_grad():
            params["quats"].data = params["quats"].data / torch.clamp(
                params["quats"].data.norm(dim=-1, keepdim=True), min=1e-12
            )

        global_step += 1
        if use_wandb and global_step % 10 == 0:
            wandb.log({"loss": loss.item()}, step=global_step)
        if use_wandb:
            _log_wandb_media(
                pred=pred,
                target=target,
                step=global_step,
                render_buffer=render_buffer,
                config=config,
            )
        if global_step % max(1, int(config.log_interval)) == 0:
            print(
                f"[step {global_step:05d}] "
                f"loss={loss.item():.4f} "
                f"ts={sample['timestamp']}"
            )

    _save_window_gaussians(
        params,
        sweep_dataset[sweep_indices[-1]],
        window_step="final",
    )
    if use_wandb:
        wandb.finish()


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scene", type=str, required=True, help="Scene name.")
    parser.add_argument(
        "--camera",
        type=str,
        default="cam_front",
        help="Camera channel (default: cam_front).",
    )
    parser.add_argument(
        "--nuscenes-version",
        type=str,
        default=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
        dest="nuscenes_version",
        help="NuScenes version (default: env or v1.0-mini).",
    )
    parser.add_argument(
        "--start-sample-index",
        type=int,
        default=0,
        dest="start_sample_index",
        help="Index in the sample-frame list to initialize SHARP Gaussians from.",
    )
    parser.add_argument(
        "--start-timestamp",
        type=int,
        default=None,
        dest="start_timestamp",
        help="Override the start timestamp (NuScenes microseconds).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        dest="steps",
        help="Training steps for the full sequence (default: one step per frame).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Base learning rate for Gaussian parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="default",
        help="Torch device string (default: auto-select).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        dest="log_interval",
        help="Log every N optimization steps.",
    )
    parser.add_argument(
        "--loss-weight-sky",
        type=float,
        default=0.0,
        dest="loss_weight_sky",
        help="Loss weight multiplier for sky pixels.",
    )
    parser.add_argument(
        "--loss-weight-movable",
        type=float,
        default=0.1,
        dest="loss_weight_movable",
        help="Loss weight multiplier for movable-object pixels.",
    )
    parser.add_argument(
        "--sharp-checkpoint",
        type=str,
        default=None,
        dest="sharp_checkpoint",
        help="Optional SHARP checkpoint path (default: download).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="geo_forge",
        dest="wandb_project",
        help="Weights & Biases project name (enables logging).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        dest="wandb_run_name",
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--wandb-render-interval",
        type=int,
        default=1000,
        dest="wandb_render_interval",
        help="Log GIF/target every N steps (default: 1000).",
    )
    parser.add_argument(
        "--wandb-render-frames",
        type=int,
        default=16,
        dest="wandb_render_frames",
        help="Number of frames in the logged GIF (default: 16).",
    )
    return parser


def parse_args() -> SequenceTrainConfig:
    parser = build_arg_parser(
        "Train a single-camera sequence with SHARP-initialized Gaussians."
    )
    args = parser.parse_args()
    loss_weights = LossWeightConfig(
        sky=float(args.loss_weight_sky),
        movable_objects=float(args.loss_weight_movable),
    )
    sharp_checkpoint = Path(args.sharp_checkpoint) if args.sharp_checkpoint else None
    return SequenceTrainConfig(
        scene=args.scene,
        camera=args.camera,
        nuscenes_version=args.nuscenes_version,
        start_sample_index=int(args.start_sample_index),
        start_timestamp=args.start_timestamp,
        steps=int(args.steps),
        lr=float(args.lr),
        device=args.device,
        log_interval=int(args.log_interval),
        loss_weights=loss_weights,
        sharp_checkpoint=sharp_checkpoint,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_render_interval=int(args.wandb_render_interval),
        wandb_render_frames=int(args.wandb_render_frames),
    )


def main() -> None:
    config = parse_args()
    train_sequence(config)


if __name__ == "__main__":
    main()

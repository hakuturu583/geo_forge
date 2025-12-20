from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import gsplat
import numpy as np
import torch
from PIL import Image
from sharp.utils.gaussians import Gaussians3D, save_ply

from geo_forge.dataset import GeoForgeDataset
from geo_forge.preprocess.sharp_util import (
    _concat_gaussians,
    _ensure_batch_gaussians,
    _load_sharp_gaussians_world,
    _flatten_gaussians,
)
from geo_forge.train.sharp_based_gs.gs_merge_prune_config import GsMergePruneConfig
from geo_forge.train.sharp_based_gs.merge_prune_strategy import MergePruneStrategy
from geo_forge.train.loss import _masked_l1_loss
from gsplat.strategy.default import DefaultStrategy


def _require_single(value: str | None, *, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required.")
    return value.strip()


def _merge_adjacent_sharp_gaussians(
    prev_meta: dict[str, object],
    curr_meta: dict[str, object],
) -> Gaussians3D | None:
    prev = _load_sharp_gaussians_world(prev_meta)
    curr = _load_sharp_gaussians_world(curr_meta)
    gaussians_list = [g for g in (prev, curr) if g is not None]
    if not gaussians_list:
        return None
    return _concat_gaussians(gaussians_list)


def _train_and_save_merged_gaussians(
    *,
    merged_gaussians: Gaussians3D,
    sweep_samples: Sequence[dict[str, object]],
    train_config: GsMergePruneConfig,
    merged_dir: Path,
    merge_idx: int,
) -> Gaussians3D:
    device_t = torch.device(
        train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    # Save raw merged Gaussians before training.
    raw_intrinsics = sweep_samples[0]["intrinsics"]
    f_px_raw = float((raw_intrinsics[0, 0] + raw_intrinsics[1, 1]) / 2.0)
    width_raw = int(sweep_samples[0]["width"])
    height_raw = int(sweep_samples[0]["height"])
    raw_ply_path = merged_dir / f"{merge_idx:04d}_raw.ply"
    save_ply(
        _ensure_batch_gaussians(_flatten_gaussians(merged_gaussians)),
        f_px_raw,
        (height_raw, width_raw),
        raw_ply_path,
    )
    print(f"Saved raw merged Gaussians to {raw_ply_path}")

    trained_gaussians = _train_gaussians_on_sweeps(
        gaussians_world=merged_gaussians,
        sweep_samples=sweep_samples,
        config=train_config,
        device=device_t,
    )

    # Save trained Gaussians to merged_gaussians/(index).ply
    first_sample = sweep_samples[0]
    intrinsics = first_sample["intrinsics"]
    f_px = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    width = int(first_sample["width"])
    height = int(first_sample["height"])
    ply_path = merged_dir / f"{merge_idx:04d}.ply"
    save_ply(
        _ensure_batch_gaussians(_flatten_gaussians(trained_gaussians)),
        f_px,
        (height, width),
        ply_path,
    )
    print(f"Saved merged Gaussians to {ply_path}")
    return trained_gaussians


def _initialize_params_from_gaussians(
    gaussians: Gaussians3D, device: torch.device
) -> dict[str, torch.nn.Parameter]:
    mean_vectors = gaussians.mean_vectors.detach()
    if mean_vectors.dim() == 3:
        mean_vectors = mean_vectors.flatten(0, 1)
    if mean_vectors.dim() != 2 or mean_vectors.shape[-1] != 3:
        raise ValueError(
            "gaussians.mean_vectors must have shape (N, 3) or (B, N, 3); "
            f"got {tuple(gaussians.mean_vectors.shape)}"
        )
    positions = mean_vectors.to(device=device, dtype=torch.float32)

    singular_values = gaussians.singular_values.detach()
    if singular_values.dim() == 3:
        singular_values = singular_values.flatten(0, 1)
    if singular_values.shape != positions.shape:
        raise ValueError(
            "gaussians.singular_values must match mean_vectors shape; "
            f"got {tuple(gaussians.singular_values.shape)}"
        )
    scales_log = (
        singular_values.to(device=device, dtype=torch.float32).clamp_min(1e-12).log()
    )

    quats = gaussians.quaternions.detach()
    if quats.dim() == 3:
        quats = quats.flatten(0, 1)
    if quats.dim() != 2 or quats.shape != (positions.shape[0], 4):
        raise ValueError(
            "gaussians.quaternions must have shape (N, 4) or (B, N, 4); "
            f"got {tuple(gaussians.quaternions.shape)}"
        )
    quats = quats.to(device=device, dtype=torch.float32)
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)

    opacities_raw = gaussians.opacities.detach()
    if opacities_raw.dim() == 3 and opacities_raw.shape[-1] == 1:
        opacities_raw = opacities_raw.squeeze(-1)
    if opacities_raw.dim() == 2:
        opacities_raw = opacities_raw.flatten(0, 1)
    if opacities_raw.dim() != 1 or opacities_raw.shape[0] != positions.shape[0]:
        raise ValueError(
            "gaussians.opacities must have shape (N,), (N, 1), (B, N), or (B, N, 1); "
            f"got {tuple(gaussians.opacities.shape)}"
        )
    opacities = torch.logit(
        opacities_raw.to(device=device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    )

    colors_raw = gaussians.colors.detach()
    if colors_raw.dim() == 3:
        colors_raw = colors_raw.flatten(0, 1)
    if colors_raw.dim() != 2 or colors_raw.shape != (positions.shape[0], 3):
        raise ValueError(
            "gaussians.colors must have shape (N, 3) or (B, N, 3); "
            f"got {tuple(gaussians.colors.shape)}"
        )
    colors = colors_raw.to(device=device, dtype=torch.float32)

    params = {
        "means": torch.nn.Parameter(positions),
        "scales": torch.nn.Parameter(scales_log),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    return params


def _render_gaussians(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    viewmat = torch.inverse(c2w)[None, ...]
    Ks = intrinsics[None, ...]
    (radii, means2d, depths, conics, _) = gsplat.rendering.fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmat,
        Ks=Ks,
        width=width,
        height=height,
        opacities=opacities,
    )

    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
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
    backgrounds = torch.zeros(1, 3, device=means.device)
    pred, _ = gsplat.rendering.rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=colors[None, ...],
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
    return pred, means2d, radii, depths


def _save_image_tensor(image: torch.Tensor, path: Path) -> None:
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image tensor shape (3, H, W); got {image.shape}")
    image_np = image.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy() * 255.0
    image_uint8 = np.clip(image_np + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def _train_gaussians_on_sweeps(
    *,
    gaussians_world: Gaussians3D,
    sweep_samples: Sequence[dict[str, object]],
    config: GsMergePruneConfig,
    device: torch.device,
) -> Gaussians3D:
    if not sweep_samples:
        raise ValueError("sweep_samples must be non-empty to train gaussians.")

    params = _initialize_params_from_gaussians(gaussians_world, device=device)
    first_sample = sweep_samples[0]
    output_dir = Path.home() / "workspace" / "geo_forge"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_image_path = output_dir / "sample_1.png"
    _save_image_tensor(first_sample["image"], sample_image_path)

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

    strategy = MergePruneStrategy(
        verbose=True,
        prune_opa=config.strategy.prune_opacity_threshold,
        prune_scale_threshold=config.strategy.prune_scale_threshold,
        backface_prune=config.backface_prune,
        grow_grad2d=config.strategy.grow_grad2d_threshold,
        refine_start_iter=config.strategy.refine_start_iter,
        refine_stop_iter=config.strategy.refine_stop_iter,
        reset_every=config.strategy.reset_every,
    )
    # strategy = DefaultStrategy(verbose=True)
    strategy_state = strategy.initialize_state()
    strategy.check_sanity(params, optimizers)

    tile_size = 16
    before_dir = output_dir / "before"
    before_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sample_idx, sample in enumerate(sweep_samples):
            pred_before, _, _, _ = _render_gaussians(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors"],
                intrinsics=sample["intrinsics"].to(device),
                c2w=sample["c2w"].to(device),
                width=int(sample["width"]),
                height=int(sample["height"]),
                tile_size=tile_size,
            )
            before_path = before_dir / f"{sample_idx:04d}.png"
            _save_image_tensor(pred_before, before_path)

    for step in range(int(config.steps)):
        for opt in optimizers.values():
            opt.zero_grad()

        sample = sweep_samples[step % len(sweep_samples)]
        image = sample["image"].to(device)  # (3, H, W)
        intrinsics = sample["intrinsics"].to(device)
        c2w = sample["c2w"].to(device)
        width = int(sample["width"])
        height = int(sample["height"])

        scales = torch.exp(params["scales"])
        opacities = torch.sigmoid(params["opacities"])

        pred, means2d, radii, depths = _render_gaussians(
            means=params["means"],
            quats=params["quats"],
            scales=scales,
            opacities=opacities,
            colors=params["colors"],
            intrinsics=intrinsics,
            c2w=c2w,
            width=width,
            height=height,
            tile_size=tile_size,
        )

        info = {
            "means2d": means2d,
            "width": width,
            "height": height,
            "n_cameras": 1,
            "radii": radii,
            "depths": depths,
            "gaussian_ids": torch.arange(
                params["means"].shape[0], device=device
            ).unsqueeze(0),
        }
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        loss = _masked_l1_loss(
            pred=pred,
            target=image,
            sample=sample,
            config=config,
            device=device,
        )
        if config.scale_anisotropy_weight > 0.0:
            scales_log = params["scales"]
            max_log = scales_log.max(dim=1).values
            min_log = scales_log.min(dim=1).values
            anisotropy = torch.relu(
                (max_log - min_log) - float(config.scale_anisotropy_log_threshold)
            )
            anisotropy_loss = (
                anisotropy.mean()
                if anisotropy.numel() > 0
                else scales_log.new_tensor(0.0)
            )
            loss = loss + float(config.scale_anisotropy_weight) * anisotropy_loss
        loss.backward()

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        if params["means"].shape[0] == 0:
            raise RuntimeError("All Gaussians were pruned/merged away.")

        for opt in optimizers.values():
            opt.step()
        with torch.no_grad():
            params["quats"].data = params["quats"].data / torch.clamp(
                params["quats"].data.norm(dim=-1, keepdim=True), min=1e-12
            )

        if (step + 1) % max(1, int(config.log_interval)) == 0:
            print(
                f"[window step {step + 1:04d}] "
                f"loss={loss.item():.4f} num_points={params['means'].shape[0]}"
            )

    after_dir = output_dir / "after"
    after_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for sample_idx, sample in enumerate(sweep_samples):
            pred_after, _, _, _ = _render_gaussians(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors"],
                intrinsics=sample["intrinsics"].to(device),
                c2w=sample["c2w"].to(device),
                width=int(sample["width"]),
                height=int(sample["height"]),
                tile_size=tile_size,
            )
            after_path = after_dir / f"{sample_idx:04d}.png"
            _save_image_tensor(pred_after, after_path)

    # Build Gaussians3D for saving/export (ensure batch dimension).
    def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    means_b = _ensure_batch(params["means"].detach())
    scales_b = _ensure_batch(torch.exp(params["scales"].detach()))
    quats_b = _ensure_batch(
        params["quats"].detach()
        / torch.clamp(params["quats"].detach().norm(dim=-1, keepdim=True), min=1e-12)
    )
    colors_b = _ensure_batch(params["colors"].detach())
    opacities_b = _ensure_batch(torch.sigmoid(params["opacities"].detach()))

    return Gaussians3D(
        mean_vectors=means_b,
        singular_values=scales_b,
        quaternions=quats_b,
        colors=colors_b,
        opacities=opacities_b,
    )


def train(
    *,
    scene: str,
    camera: str,
    nuscenes_version: str = os.getenv("NUSCENES_VERSION", "v1.0-mini"),
    config: GsMergePruneConfig | None = None,
) -> None:
    """
    Prepare a single-camera sequence dataset in two variants.

    This creates two datasets with the same ``scene``/``camera`` filters:
    - ``sample_dataset``: keyframes only (NuScenes synchronized samples)
    - ``sweep_dataset``: includes intermediate sweep frames as well

    Args:
        scene: Scene directory name under ``GEOFORGE_DATASET_ROOT`` (e.g., "scene-0061").
        camera: Camera channel name (e.g., "cam_front" or "CAM_FRONT").
        nuscenes_version: NuScenes version string (defaults to ``$NUSCENES_VERSION`` or
            ``v1.0-mini``).
    """
    train_config = config or GsMergePruneConfig()
    scene_name = _require_single(scene, name="scene")
    camera_name = _require_single(camera, name="camera")
    camera_dir = camera_name.lower()
    scene_filter: Sequence[str] = [scene_name]
    camera_filter: Sequence[str] = [camera_dir]
    dataset_root_env = os.getenv("GEOFORGE_DATASET_ROOT")
    if not dataset_root_env:
        raise EnvironmentError(
            "GEOFORGE_DATASET_ROOT must be set to save merged Gaussians."
        )
    dataset_root = Path(dataset_root_env).expanduser()
    merged_dir = dataset_root / scene_name / camera_dir / "merged_gaussians"
    merged_dir.mkdir(parents=True, exist_ok=True)

    sample_dataset = GeoForgeDataset(
        version=nuscenes_version,
        scene_filter=scene_filter,
        camera_filter=camera_filter,
        only_sample_frames=True,
    )
    sweep_dataset = GeoForgeDataset(
        version=nuscenes_version,
        scene_filter=scene_filter,
        camera_filter=camera_filter,
        only_sample_frames=False,
    )

    sample_metas = sorted(
        sample_dataset.samples,
        key=lambda sample: int(sample["timestamp"]),
    )
    if len(sample_metas) < 2:
        print("Not enough sample frames to merge Gaussians.")
        return

    merge_idx = 0
    prev_meta = sample_metas[0]
    curr_meta = sample_metas[1]
    prev_ts = int(prev_meta["timestamp"])
    curr_ts = int(curr_meta["timestamp"])
    start_ts = min(prev_ts, curr_ts)
    end_ts = max(prev_ts, curr_ts)

    merged_gaussians = _merge_adjacent_sharp_gaussians(prev_meta, curr_meta)
    if merged_gaussians is None:
        print("No Gaussians found for the initial sample pair.")
        return

    sweep_samples = sweep_dataset.get_samples_between(
        start_ts,
        end_ts,
        inclusive=True,
    )
    if not sweep_samples:
        print("No sweep samples found for the initial sample pair.")
        return

    print(
        "Training merged SHARP Gaussians:",
        f"prev_ts={prev_ts}",
        f"curr_ts={curr_ts}",
        f"sweeps={len(sweep_samples)}",
    )
    merged_gaussians = _train_and_save_merged_gaussians(
        merged_gaussians=merged_gaussians,
        sweep_samples=sweep_samples,
        train_config=train_config,
        merged_dir=merged_dir,
        merge_idx=merge_idx,
    )

    # last_meta = curr_meta
    # for next_meta in sample_metas[2:]:
    #     prev_ts = int(last_meta["timestamp"])
    #     curr_ts = int(next_meta["timestamp"])
    #     start_ts = min(prev_ts, curr_ts)
    #     end_ts = max(prev_ts, curr_ts)

    #     next_gaussians = _load_sharp_gaussians_world(next_meta)
    #     gaussians_list = [
    #         g for g in (merged_gaussians, next_gaussians) if g is not None
    #     ]
    #     if not gaussians_list:
    #         last_meta = next_meta
    #         continue

    #     merged_gaussians = _concat_gaussians(gaussians_list)
    #     sweep_samples = sweep_dataset.get_samples_between(
    #         start_ts,
    #         end_ts,
    #         inclusive=True,
    #     )
    #     if not sweep_samples:
    #         last_meta = next_meta
    #         continue

    #     merge_idx += 1
    #     print(
    #         "Training merged SHARP Gaussians:",
    #         f"prev_ts={prev_ts}",
    #         f"curr_ts={curr_ts}",
    #         f"sweeps={len(sweep_samples)}",
    #     )
    #     merged_gaussians = _train_and_save_merged_gaussians(
    #         merged_gaussians=merged_gaussians,
    #         sweep_samples=sweep_samples,
    #         train_config=train_config,
    #         merged_dir=merged_dir,
    #         merge_idx=merge_idx,
    #     )
    #     last_meta = next_meta

    # print(
    #     "Finished merge-prune training windowing.",
    #     f"sample_frames={len(sample_dataset)}",
    #     f"sweep_frames={len(sweep_dataset)}",
    # )


def train_one_camera_sequence(
    *,
    scene: str,
    camera: str,
    nuscenes_version: str = os.getenv("NUSCENES_VERSION", "v1.0-mini"),
) -> None:
    """
    Alias for :func:`train`.
    """
    return train(
        scene=scene,
        camera=camera,
        nuscenes_version=nuscenes_version,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample-only and sweep-inclusive datasets for one scene/camera."
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene directory name under GEOFORGE_DATASET_ROOT (e.g., scene-0061).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera channel (e.g., cam_front or CAM_FRONT).",
    )
    parser.add_argument(
        "--nuscenes-version",
        type=str,
        default=os.getenv("NUSCENES_VERSION", "v1.0-mini"),
        dest="nuscenes_version",
        help="Optional NuScenes version override (e.g., v1.0-mini).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML path for GsMergePruneConfig (steps/lr/strategy/loss weights).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GsMergePruneConfig.from_yaml(args.config) if args.config else None
    train(
        scene=args.scene,
        camera=args.camera,
        nuscenes_version=args.nuscenes_version,
        config=config,
    )


if __name__ == "__main__":
    main()

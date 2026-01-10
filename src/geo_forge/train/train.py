from __future__ import annotations

import math
from datetime import datetime
import copy
import types
import time
from collections.abc import Iterable as AbcIterable, Sequence as AbcSequence
from typing import Any, Iterable, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np
from dataclasses import fields, is_dataclass
import gsplat
import torch
import wandb
from dotenv import load_dotenv
from gsplat.strategy import DefaultStrategy
from gsplat.strategy import default as gs_default
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf
from sharp.utils.gaussians import Gaussians3D
from tqdm import tqdm

from geo_forge.dataset import GeoForgeDataset
from geo_forge.train.gs_train_config import GsTrainConfig, LodConfig
from geo_forge.train.loss import Loss
from geo_forge.preprocess.sharp_util import load_gaussians_from_sharp_ply


load_dotenv()


def _initialize_params(
    dataset: GeoForgeDataset,
    config: GsTrainConfig,
    device: torch.device,
    init_gaussians: Gaussians3D | None = None,
) -> tuple[dict[str, torch.nn.Parameter], torch.Tensor]:
    """
    Build the trainable Gaussian parameter tensors, optionally seeding from
    a SHARP ``Gaussians3D`` prediction.
    """
    if init_gaussians is not None:
        mean_vectors = init_gaussians.mean_vectors.detach()
        if mean_vectors.dim() == 3:
            mean_vectors = mean_vectors.flatten(0, 1)
        if mean_vectors.dim() != 2 or mean_vectors.shape[-1] != 3:
            raise ValueError(
                "init_gaussians.mean_vectors must have shape (N, 3) or (B, N, 3); "
                f"got {tuple(init_gaussians.mean_vectors.shape)}"
            )
        num_init = int(mean_vectors.shape[0])
        if num_init == 0:
            raise ValueError(
                "init_gaussians contains zero Gaussians; pass None to randomly initialize."
            )

        if num_init != config.num_gaussians:
            print(
                f"Initializing {num_init} Gaussians from SHARP Gaussians3D "
                f"(config.num_gaussians={config.num_gaussians})."
            )

        positions = mean_vectors.to(device=device, dtype=torch.float32)

        singular_values = init_gaussians.singular_values.detach()
        if singular_values.dim() == 3:
            singular_values = singular_values.flatten(0, 1)
        if singular_values.shape != positions.shape:
            raise ValueError(
                "init_gaussians.singular_values must match mean_vectors shape; "
                f"got {tuple(init_gaussians.singular_values.shape)}"
            )
        scales_log = (
            singular_values.to(device=device, dtype=torch.float32)
            .clamp_min(1e-12)
            .log()
        )

        quats = init_gaussians.quaternions.detach()
        if quats.dim() == 3:
            quats = quats.flatten(0, 1)
        if quats.dim() != 2 or quats.shape != (num_init, 4):
            raise ValueError(
                "init_gaussians.quaternions must have shape (N, 4) or (B, N, 4); "
                f"got {tuple(init_gaussians.quaternions.shape)}"
            )
        quats = quats.to(device=device, dtype=torch.float32)

        opacities_raw = init_gaussians.opacities.detach()
        if opacities_raw.dim() == 3 and opacities_raw.shape[-1] == 1:
            opacities_raw = opacities_raw.squeeze(-1)
        if opacities_raw.dim() == 2:
            opacities_raw = opacities_raw.flatten(0, 1)
        if opacities_raw.dim() != 1 or opacities_raw.shape[0] != num_init:
            raise ValueError(
                "init_gaussians.opacities must have shape (N,), (N, 1), (B, N), or (B, N, 1); "
                f"got {tuple(init_gaussians.opacities.shape)}"
            )
        opacities = torch.logit(
            opacities_raw.to(device=device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        )

        colors_raw = init_gaussians.colors.detach()
        if colors_raw.dim() == 3:
            colors_raw = colors_raw.flatten(0, 1)
        if colors_raw.dim() != 2 or colors_raw.shape != (num_init, 3):
            raise ValueError(
                "init_gaussians.colors must have shape (N, 3) or (B, N, 3); "
                f"got {tuple(init_gaussians.colors.shape)}"
            )
        colors = colors_raw.to(device=device, dtype=torch.float32)
    else:
        num_init = config.num_gaussians
        if num_init <= 0:
            raise ValueError("num_gaussians must be positive.")

        # Initialize scales in log-space with small values so the strategy's scale-based
        # pruning does not immediately drop every Gaussian after the first reset.
        scale_base = 0.01
        scale_jitter = 0.005
        init_scales = torch.full((num_init, 3), scale_base, device=device)
        init_scales += scale_jitter * torch.rand_like(init_scales)

        positions = dataset.get_init_gaussian_means(num_samples=num_init).to(device)
        scales_log = init_scales.log()
        quats = torch.rand((num_init, 4), device=device)
        opacities = torch.rand((num_init,), device=device)
        colors = torch.rand((num_init, 3), device=device)

    init_means = positions.detach().clone()
    params = {
        "means": torch.nn.Parameter(positions),
        "scales": torch.nn.Parameter(scales_log),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    params["quats"].data = params["quats"].data / params["quats"].data.norm(
        dim=-1, keepdim=True
    )
    return params, init_means


def _load_initial_gaussians(
    dataset: GeoForgeDataset, config: GsTrainConfig
) -> Gaussians3D | None:
    scene_name: str | None
    if config.scenes:
        if len(config.scenes) != 1:
            print(
                "Multiple scenes configured; skipping initial_gaussians.ply initialization."
            )
            return None
        scene_name = config.scenes[0]
    else:
        scene_name = dataset.samples[0]["scene"] if dataset.samples else None

    if not scene_name:
        return None

    ply_path = dataset.dataset_root / scene_name / "initial_gaussians.ply"
    if not ply_path.exists():
        print(f"No initial_gaussians.ply found at {ply_path}; using random init.")
        return None

    gaussians, _, _, _, _ = load_gaussians_from_sharp_ply(ply_path)
    mean_vectors = gaussians.mean_vectors
    if mean_vectors.dim() == 3:
        num_points = mean_vectors.shape[0] * mean_vectors.shape[1]
    elif mean_vectors.dim() == 2:
        num_points = mean_vectors.shape[0]
    else:
        num_points = 0
    print(f"Loaded initial_gaussians.ply from {ply_path} ({num_points} Gaussians).")
    return gaussians


def train_gaussian_splatting(
    dataset: GeoForgeDataset,
    config: GsTrainConfig,
    init_gaussians: Gaussians3D | None = None,
    lod_state: dict[str, object] | None = None,
) -> None:
    """
    Lightweight training loop that optimizes Gaussian parameters against ROSE frames.

    Args:
        dataset: GeoForgeDataset providing frames and poses.
        config: Training hyperparameters.
        init_gaussians: Optional SHARP Gaussians3D prediction to seed parameters.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; nothing to train on.")

    device_t = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    base_lr = config.lr_config.base_lr
    lr_config = config.lr_config

    init_start = time.perf_counter()
    if init_gaussians is None:
        init_gaussians = _load_initial_gaussians(dataset, config)

    params, init_means = _initialize_params(
        dataset=dataset,
        config=config,
        device=device_t,
        init_gaussians=init_gaussians,
    )
    init_elapsed = time.perf_counter() - init_start
    print(f"[train] params initialized in {init_elapsed:.2f}s")

    optimizers = {
        "means": torch.optim.Adam(
            [params["means"]],
            lr=base_lr * lr_config.means.value,
            eps=lr_config.means.eps,
        ),
        "scales": torch.optim.Adam(
            [params["scales"]],
            lr=base_lr * lr_config.scales.value,
            eps=lr_config.scales.eps,
        ),
        "quats": torch.optim.Adam(
            [params["quats"]],
            lr=base_lr * lr_config.quats.value,
            eps=lr_config.quats.eps,
        ),
        "opacities": torch.optim.Adam(
            [params["opacities"]],
            lr=base_lr * lr_config.opacities.value,
            eps=lr_config.opacities.eps,
        ),
        "colors": torch.optim.Adam(
            [params["colors"]],
            lr=base_lr * lr_config.colors.value,
            eps=lr_config.colors.eps,
        ),
    }
    strategy = DefaultStrategy(
        verbose=True,
        prune_opa=config.strategy.prune_opacity_threshold,
        grow_grad2d=config.strategy.grow_grad2d_threshold,
        refine_start_iter=config.strategy.refine_start_iter,
        refine_stop_iter=config.strategy.refine_stop_iter,
        reset_every=config.strategy.reset_every,
        refine_every=config.strategy.refine_every,
    )
    strategy_state = strategy.initialize_state()
    strategy.check_sanity(params, optimizers)
    print("[train] strategy initialized")

    use_wandb = bool(config.wandb_project)
    if use_wandb and wandb.run is None:
        run_name = (
            config.wandb_run_name
            if config.wandb_run_name is not None
            else datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config=_build_wandb_config(config),
        )

    render_interval = (
        config.render_interval
        if config.render_interval is not None
        else config.log_interval
    )
    loss_fn = Loss(config.loss_weights).to(device_t)
    print("[train] entering training loop")

    for step in tqdm(range(config.steps), desc="train", unit="step"):
        for opt in optimizers.values():
            opt.zero_grad()

        sample = dataset[step % len(dataset)]
        image = sample["image"].to(device_t)  # (3, H, W)
        intrinsics = sample["intrinsics"].to(device_t)
        c2w = sample["c2w"].to(device_t)
        width = int(sample["width"])
        height = int(sample["height"])

        # Activate parameters for rendering; keep raw tensors (log-scales/logits)
        # for optimization and pruning heuristics.
        scales = torch.exp(params["scales"])
        opacities = torch.sigmoid(params["opacities"])

        viewmat = torch.inverse(c2w)[None, ...]
        Ks = intrinsics[None, ...]
        if config.render_packed:
            packed_outputs = gsplat.rendering.fully_fused_projection(
                means=params["means"],
                covars=None,
                quats=params["quats"],
                scales=scales,
                viewmats=viewmat,
                Ks=Ks,
                width=width,
                height=height,
                opacities=opacities,
                packed=True,
            )
            if len(packed_outputs) == 7:
                (
                    camera_ids,
                    gaussian_ids,
                    radii,
                    means2d,
                    depths,
                    conics,
                    _,
                ) = packed_outputs
            elif len(packed_outputs) == 8:
                (
                    _batch_ids,
                    camera_ids,
                    gaussian_ids,
                    radii,
                    means2d,
                    depths,
                    conics,
                    _,
                ) = packed_outputs
            else:
                raise ValueError(
                    "Unexpected packed projection output count: "
                    f"{len(packed_outputs)}"
                )
            max_id = params["means"].shape[0]
            valid_ids = (gaussian_ids >= 0) & (gaussian_ids < max_id)
            if not torch.all(valid_ids):
                invalid_count = int((~valid_ids).sum().item())
                print(
                    f"[train] warning: filtered {invalid_count} invalid packed gaussian_ids"
                )
                gaussian_ids = gaussian_ids[valid_ids]
                camera_ids = camera_ids[valid_ids] if camera_ids is not None else None
                radii = radii[valid_ids]
                means2d = means2d[valid_ids]
                depths = depths[valid_ids]
                conics = conics[valid_ids]
            colors = params["colors"][gaussian_ids]
            opacities_render = opacities[gaussian_ids]
        else:
            (
                radii,
                means2d,
                depths,
                conics,
                _,
            ) = gsplat.rendering.fully_fused_projection(
                means=params["means"],
                covars=None,
                quats=params["quats"],
                scales=scales,
                viewmats=viewmat,
                Ks=Ks,
                width=width,
                height=height,
                opacities=opacities,
                packed=False,
            )
            camera_ids = None
            gaussian_ids = None
            colors = params["colors"][None, ...]
            opacities_render = opacities[None, ...]

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
            packed=config.render_packed,
            n_images=1 if config.render_packed else None,
            image_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
        isect_offsets = gsplat.rendering.isect_offset_encode(
            isect_ids=isect_ids,
            n_images=1,
            tile_width=tile_width,
            tile_height=tile_height,
        )
        if config.render_packed:
            backgrounds = torch.zeros(3, device=device_t)
        else:
            backgrounds = torch.zeros(1, 3, device=device_t)
        pred, _ = gsplat.rendering.rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=colors,
            opacities=opacities_render,
            image_width=width,
            image_height=height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=backgrounds,
            packed=config.render_packed,
            absgrad=False,
        )
        if pred.dim() == 5:
            pred = pred.permute(0, 1, 4, 2, 3)[0, 0]
        elif pred.dim() == 4:
            pred = pred.permute(0, 3, 1, 2)[0]
        elif pred.dim() == 3:
            pred = pred.permute(2, 0, 1)

        if config.render_packed:
            gaussian_ids_info = gaussian_ids
        else:
            gaussian_ids_info = torch.arange(
                params["means"].shape[0], device=device_t
            ).unsqueeze(0)
        info = {
            "means2d": means2d,
            "width": width,
            "height": height,
            "n_cameras": 1,
            "radii": radii,
            "depths": depths,
            "gaussian_ids": gaussian_ids_info,
        }
        far_mask = None
        if lod_state is not None:
            far_mask = _update_lod_mask(
                lod_state=lod_state,
                means=params["means"],
                device=device_t,
                step=step,
                config=config.lod,
            )
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        loss_sample = dict(sample)
        loss_sample["gaussian_means"] = params["means"]
        loss_sample["gaussian_quats"] = params["quats"]
        loss_sample["init_gaussian_means"] = init_means.to(device_t)
        loss_sample["gaussian_opacities"] = opacities
        loss_sample["gaussian_scales"] = scales
        loss, loss_components = loss_fn.compute(
            pred=pred,
            target=image,
            sample=loss_sample,
            step=step,
            total_steps=config.steps,
        )

        loss.backward()

        if far_mask is not None:
            _apply_lod_grad_scale(
                info=info,
                far_mask=far_mask,
                packed=config.render_packed,
                scale=config.lod.far_grad_scale,
            )
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
            packed=config.render_packed,
        )
        if far_mask is not None:
            _prune_far_gaussians(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                far_mask=far_mask,
                config=config,
            )

        if params["means"].shape[0] == 0:
            raise RuntimeError(
                "All Gaussians were pruned. Initialize with smaller scales or relax "
                "the pruning thresholds to avoid an empty set."
            )

        for opt in optimizers.values():
            opt.step()
        with torch.no_grad():
            params["quats"].data = params["quats"].data / torch.clamp(
                params["quats"].data.norm(dim=-1, keepdim=True), min=1e-12
            )

        if use_wandb:
            metrics = loss_fn.build_wandb_log(loss, loss_components)
            metrics["gaussians/count"] = float(params["means"].shape[0])
            wandb.log(metrics, step=step + 1)

        if (step + 1) % config.log_interval == 0:
            scene = sample["scene"]
            camera = sample["camera"]
            timestamp = sample["timestamp"]
            print(
                f"[step {step + 1:04d}] "
                f"loss={loss.item():.4f} "
                f"scene={scene} cam={camera} ts={timestamp} "
                f"num_points={params['means'].shape[0]}"
            )

        if use_wandb and render_interval and (step + 1) % render_interval == 0:
            # Log rendered prediction for quick qualitative checks.
            pred_img = pred.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
            target_img = image.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
            wandb.log(
                {
                    "render/prediction": wandb.Image(pred_img),
                    "render/target": wandb.Image(target_img),
                },
                step=step + 1,
            )


def _set_by_path(root: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    cursor = root
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = value


def _unwrap_optional(tp: Any) -> Any:
    origin = get_origin(tp)
    if origin is None:
        return tp
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(tp) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _is_list_type(tp: Any) -> bool:
    tp = _unwrap_optional(tp)
    origin = get_origin(tp)
    if origin is None:
        return False
    return origin in (list, Iterable, Sequence, AbcIterable, AbcSequence)


def _collect_sweep_params(
    raw: dict[str, object],
    schema: type,
    *,
    prefix: str = "",
) -> dict[str, list[object]]:
    sweep: dict[str, list[object]] = {}
    if not is_dataclass(schema):
        return sweep
    type_hints = get_type_hints(schema)
    for field in fields(schema):
        name = field.name
        if name not in raw:
            continue
        value = raw[name]
        field_type = _unwrap_optional(type_hints.get(name, field.type))
        key = f"{prefix}{name}"
        if isinstance(value, dict) and is_dataclass(field_type):
            sweep.update(_collect_sweep_params(value, field_type, prefix=f"{key}."))
            continue
        if isinstance(value, list) and not _is_list_type(field_type):
            if not value:
                raise ValueError(f"Sweep list for {key} must be non-empty.")
            if not all(isinstance(item, (int, float, str, bool)) for item in value):
                raise ValueError(
                    f"Sweep list for {key} must contain primitive values only."
                )
            sweep[key] = value
    return sweep


def _build_wandb_config(config: GsTrainConfig) -> dict[str, object]:
    lr_config = config.lr_config
    lod_config = config.lod
    return {
        "num_steps": config.steps,
        "num_gaussians": config.num_gaussians,
        "base_lr": lr_config.base_lr,
        "lr_multipliers": {
            "means": lr_config.means.value,
            "scales": lr_config.scales.value,
            "quats": lr_config.quats.value,
            "opacities": lr_config.opacities.value,
            "colors": lr_config.colors.value,
        },
        "lr_eps": {
            "means": lr_config.means.eps,
            "scales": lr_config.scales.eps,
            "quats": lr_config.quats.eps,
            "opacities": lr_config.opacities.eps,
            "colors": lr_config.colors.eps,
        },
        "log_every": config.log_interval,
        "log_render_every": config.render_interval,
        "render_packed": config.render_packed,
        "prune_opacity_threshold": config.strategy.prune_opacity_threshold,
        "grow_grad2d_threshold": config.strategy.grow_grad2d_threshold,
        "refine_start_iter": config.strategy.refine_start_iter,
        "refine_stop_iter": config.strategy.refine_stop_iter,
        "reset_every": config.strategy.reset_every,
        "loss_opacity_weight": config.loss_weights.opacity.weight,
        "loss_scale_weight": config.loss_weights.scale.weight,
        "loss_scale_max": config.loss_weights.scale.max_scale,
        "loss_anisotropy_weight": config.loss_weights.anisotropy.weight,
        "loss_anisotropy_max_ratio": config.loss_weights.anisotropy.max_ratio,
        "lod_enabled": lod_config.enabled,
        "lod_distance_threshold": lod_config.distance_threshold,
        "lod_update_interval": lod_config.update_interval,
        "lod_query_chunk_size": lod_config.query_chunk_size,
        "lod_far_grad_scale": lod_config.far_grad_scale,
        "lod_far_prune_opacity_multiplier": lod_config.far_prune_opacity_multiplier,
        "lod_far_prune_scale_multiplier": lod_config.far_prune_scale_multiplier,
    }


def _build_lod_state(
    dataset: GeoForgeDataset, config: LodConfig
) -> dict[str, object] | None:
    if not config.enabled:
        return None
    kdtree, _ = dataset.build_camera_pose_kdtree()
    if kdtree.n == 0:
        raise RuntimeError("LOD requested but no camera poses are available.")
    return {
        "kdtree": kdtree,
        "far_mask": None,
        "last_update_step": -1,
        "last_count": -1,
    }


def _update_lod_mask(
    *,
    lod_state: dict[str, object],
    means: torch.Tensor,
    device: torch.device,
    step: int,
    config: LodConfig,
) -> torch.Tensor | None:
    if means.shape[0] != int(lod_state["last_count"]):
        lod_state["last_update_step"] = -1
    if (
        step - int(lod_state["last_update_step"]) < config.update_interval
        and lod_state.get("far_mask") is not None
    ):
        return lod_state.get("far_mask")

    kdtree = lod_state["kdtree"]
    chunk_size = max(1, int(config.query_chunk_size))
    means_np = means.detach().cpu().numpy()
    distances = np.empty((means_np.shape[0],), dtype=np.float32)
    for start in range(0, means_np.shape[0], chunk_size):
        end = min(start + chunk_size, means_np.shape[0])
        dist_chunk, _ = kdtree.query(means_np[start:end], k=1, workers=-1)
        distances[start:end] = dist_chunk.astype(np.float32, copy=False)
    far_mask = torch.from_numpy(distances > float(config.distance_threshold)).to(
        device=device
    )
    lod_state["far_mask"] = far_mask
    lod_state["last_update_step"] = step
    lod_state["last_count"] = means.shape[0]
    return far_mask


def _apply_lod_grad_scale(
    *,
    info: dict[str, object],
    far_mask: torch.Tensor,
    packed: bool,
    scale: float,
) -> None:
    means2d = info.get("means2d")
    if not isinstance(means2d, torch.Tensor) or means2d.grad is None:
        return
    if packed:
        gaussian_ids = info.get("gaussian_ids")
        if not isinstance(gaussian_ids, torch.Tensor):
            return
        if gaussian_ids.max().item() >= far_mask.shape[0]:
            return
        mask = far_mask[gaussian_ids]
        means2d.grad[mask] *= scale
    else:
        max_count = means2d.grad.shape[1]
        if far_mask.shape[0] < max_count:
            mask = torch.zeros(
                (max_count,), device=far_mask.device, dtype=far_mask.dtype
            )
            mask[: far_mask.shape[0]] = far_mask
        else:
            mask = far_mask[:max_count]
        means2d.grad[:, mask, :] *= scale


def _prune_far_gaussians(
    *,
    params: dict[str, torch.nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    state: dict[str, object],
    far_mask: torch.Tensor,
    config: GsTrainConfig,
) -> int:
    if far_mask.shape[0] != params["means"].shape[0]:
        return 0
    opacities = torch.sigmoid(params["opacities"]).flatten()
    scales = torch.exp(params["scales"])
    prune_opa = (
        config.strategy.prune_opacity_threshold
        * config.lod.far_prune_opacity_multiplier
    )
    prune_scale = (
        config.strategy.prune_scale_threshold * config.lod.far_prune_scale_multiplier
    )
    too_transparent = opacities < prune_opa
    too_large = scales.max(dim=-1).values > prune_scale
    is_prune = far_mask & (too_transparent | too_large)
    if is_prune.any():
        gs_default.remove(
            params=params, optimizers=optimizers, state=state, mask=is_prune
        )
    return int(is_prune.sum().item())


def _run_training(config: GsTrainConfig) -> None:
    start = time.perf_counter()
    dataset = GeoForgeDataset(
        scene_filter=config.scenes,
        camera_filter=config.cameras,
    )
    elapsed = time.perf_counter() - start
    print(f"[train] dataset ready in {elapsed:.2f}s (samples={len(dataset)})")
    lod_state = _build_lod_state(dataset, config.lod)
    train_gaussian_splatting(dataset, config=config, lod_state=lod_state)


def _run_wandb_sweep(
    base_raw: dict[str, object],
    sweep_params: dict[str, list[object]],
) -> None:
    project = base_raw.get("wandb_project")
    if not project:
        raise ValueError("wandb_project must be set when using sweep parameters.")

    sweep_config = {
        "method": "grid",
        "parameters": {
            name: {"values": values} for name, values in sweep_params.items()
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=str(project))

    def _train_once() -> None:
        run_name = base_raw.get("wandb_run_name")
        wandb.init(
            project=str(project),
            name=str(run_name) if run_name else None,
        )
        raw = copy.deepcopy(base_raw)
        for key in sweep_params:
            if key in wandb.config:
                _set_by_path(raw, key, wandb.config[key])
        config = GsTrainConfig.from_raw(raw)
        wandb.config.update(_build_wandb_config(config), allow_val_change=True)
        _run_training(config)
        wandb.finish()

    wandb.agent(sweep_id, function=_train_once)


def _strip_hydra(raw: dict[str, object]) -> dict[str, object]:
    cleaned = dict(raw)
    cleaned.pop("hydra", None)
    return cleaned


@hydra_main(
    version_base=None, config_path="../../../configs", config_name="gs_train_example"
)
def main(cfg: DictConfig) -> None:
    start = time.perf_counter()
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping.")
    base_raw = _strip_hydra(raw)
    elapsed = time.perf_counter() - start
    print(f"[train] config resolved in {elapsed:.2f}s")
    sweep_params = _collect_sweep_params(base_raw, GsTrainConfig)
    if sweep_params:
        _run_wandb_sweep(base_raw, sweep_params)
    else:
        config = GsTrainConfig.from_raw(base_raw)
        _run_training(config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
from datetime import datetime

import gsplat
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from gsplat.strategy import DefaultStrategy

from geo_forge.train.dataset import RoseNuScenesDataset
from geo_forge.train.gs_train_config import GsTrainConfig


load_dotenv()


def train_gaussian_splatting(
    dataset: RoseNuScenesDataset,
    config: GsTrainConfig,
) -> None:
    """
    Lightweight training loop that optimizes Gaussian parameters against ROSE frames.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; nothing to train on.")

    device_t = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    num_init = config.num_gaussians
    base_lr = config.lr

    # Initialize scales in log-space with small values so the strategy's scale-based
    # pruning does not immediately drop every Gaussian after the first reset.
    scale_base = 0.01
    scale_jitter = 0.005
    init_scales = torch.full((num_init, 3), scale_base, device=device_t)
    init_scales += scale_jitter * torch.rand_like(init_scales)

    params = {
        "means": torch.nn.Parameter(
            dataset.get_init_gaussian_means(num_samples=num_init).to(device_t)
        ),
        "scales": torch.nn.Parameter(init_scales.log()),
        "quats": torch.nn.Parameter(torch.rand((num_init, 4), device=device_t)),
        "opacities": torch.nn.Parameter(torch.rand((num_init,), device=device_t)),
        "colors": torch.nn.Parameter(torch.rand((num_init, 3), device=device_t)),
    }
    params["quats"].data = params["quats"].data / params["quats"].data.norm(
        dim=-1, keepdim=True
    )

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
    if use_wandb:
        run_name = (
            config.wandb_run_name
            if config.wandb_run_name is not None
            else datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "num_steps": config.steps,
                "num_gaussians": config.num_gaussians,
                "lr": config.lr,
                "log_every": config.log_interval,
                "log_render_every": config.render_interval,
                "prune_opacity_threshold": config.strategy.prune_opacity_threshold,
                "grow_grad2d_threshold": config.strategy.grow_grad2d_threshold,
                "refine_start_iter": config.strategy.refine_start_iter,
                "refine_stop_iter": config.strategy.refine_stop_iter,
                "reset_every": config.strategy.reset_every,
            },
        )

    render_interval = (
        config.render_interval
        if config.render_interval is not None
        else config.log_interval
    )

    for step in range(config.steps):
        for opt in optimizers.values():
            opt.zero_grad()

        sample = dataset[step % len(dataset)]
        image = sample["image"].to(device_t)  # (3, H, W)
        intrinsics = sample["intrinsics"].to(device_t)
        c2w = sample["c2w"].to(device_t)
        width = int(sample["width"])
        height = int(sample["height"])
        sky_mask = sample.get("sky_mask")
        object_mask = sample.get("object_mask")
        loss_weights = torch.ones((1, height, width), device=device_t)
        if sky_mask is not None:
            loss_weights = torch.where(
                sky_mask.to(device_t).unsqueeze(0).bool(),
                torch.tensor(config.sky_loss_weight, device=device_t),
                loss_weights,
            )
        if object_mask is not None:
            obj_mask = object_mask.to(device_t).unsqueeze(0)
            loss_weights = torch.where(
                obj_mask.bool(),
                torch.tensor(config.movable_object_loss_weight, device=device_t),
                loss_weights,
            )

        # Activate parameters for rendering; keep raw tensors (log-scales/logits)
        # for optimization and pruning heuristics.
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
        backgrounds = torch.zeros(1, 3, device=device_t)
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
            "gaussian_ids": torch.arange(
                params["means"].shape[0], device=device_t
            ).unsqueeze(0),
        }
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        loss_map = F.l1_loss(pred, image, reduction="none")
        weights = loss_weights.expand_as(loss_map).to(loss_map.dtype)
        weight_sum = weights.sum()
        if weight_sum.item() > 0:
            loss = (loss_map * weights).sum() / weight_sum
        else:
            loss = loss_map.new_tensor(0.0)

        loss.backward()

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
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
            wandb.log(
                {"loss": loss.item(), "num_points": params["means"].shape[0]},
                step=step + 1,
            )

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Gaussian splatting demo using ROSE outputs and NuScenes poses."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML file containing GsTrainConfig values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GsTrainConfig.from_yaml(args.config)
    dataset = RoseNuScenesDataset(
        scene_filter=config.scenes,
        camera_filter=config.cameras,
    )
    train_gaussian_splatting(dataset, config=config)


if __name__ == "__main__":
    main()

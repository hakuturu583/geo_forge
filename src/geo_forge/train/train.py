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
    means = torch.rand((num_init, 3), device=device_t, requires_grad=True)
    scales = torch.rand((num_init, 3), device=device_t, requires_grad=True)
    quats = torch.rand((num_init, 4), device=device_t, requires_grad=True)
    quats.data = quats.data / quats.data.norm(dim=-1, keepdim=True)
    opacities = torch.rand((num_init,), device=device_t, requires_grad=True)
    colors = torch.rand((num_init, 3), device=device_t, requires_grad=True)

    params_list = [
        {"params": [means], "lr": base_lr * 0.032, "name": "means"},
        {"params": [scales], "lr": base_lr * 1.0, "name": "scales"},
        {"params": [quats], "lr": base_lr * 0.2, "name": "quats"},
        {"params": [opacities], "lr": base_lr * 10.0, "name": "opacities"},
        {"params": [colors], "lr": base_lr * 0.5, "name": "colors"},
    ]
    optimizer = torch.optim.Adam(params_list, eps=1e-15)
    strategy = DefaultStrategy(
        verbose=True,
        prune_opa=0.005,
        grow_grad2d=0.0002,
        refine_start_iter=500,
        refine_stop_iter=15000,
        reset_every=3000,
    )

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
            },
        )

    render_interval = (
        config.render_interval
        if config.render_interval is not None
        else config.log_interval
    )

    for step in range(config.steps):
        sample = dataset[step % len(dataset)]
        image = sample["image"].to(device_t)  # (3, H, W)
        intrinsics = sample["intrinsics"].to(device_t)
        c2w = sample["c2w"].to(device_t)
        width = int(sample["width"])
        height = int(sample["height"])

        viewmat = torch.inverse(c2w)[None, ...]
        Ks = intrinsics[None, ...]
        strategy.check_sanity(
            {"means": means, "scales": scales, "quats": quats, "opacities": opacities}
        )
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
        means2d.retain_grad()

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

        loss = F.mse_loss(pred, image)

        optimizer.zero_grad()
        loss.backward()

        strategy.step_post_backward(
            params=params_list,
            optimizers=[optimizer],
            state={
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "radii": radii,
                "xys": means2d,
                "xys.grad": means2d.grad,
                "step": step,
            },
        )

        optimizer.step()

        means = params_list[0]["params"][0]
        scales = params_list[1]["params"][0]
        quats = params_list[2]["params"][0]
        opacities = params_list[3]["params"][0]
        colors = params_list[4]["params"][0]
        with torch.no_grad():
            quats.data = quats.data / torch.clamp(
                quats.data.norm(dim=-1, keepdim=True), min=1e-12
            )

        if use_wandb:
            wandb.log(
                {"loss": loss.item(), "num_points": means.shape[0]}, step=step + 1
            )

        if (step + 1) % config.log_interval == 0:
            scene = sample["scene"]
            camera = sample["camera"]
            timestamp = sample["timestamp"]
            print(
                f"[step {step + 1:04d}] "
                f"loss={loss.item():.4f} "
                f"scene={scene} cam={camera} ts={timestamp} "
                f"num_points={means.shape[0]}"
            )

        if use_wandb and render_interval and (step + 1) % render_interval == 0:
            pass


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

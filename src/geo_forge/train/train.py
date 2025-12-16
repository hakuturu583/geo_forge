from __future__ import annotations

import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv

from geo_forge.train.dataset import RoseNuScenesDataset
from geo_forge.train.gs_train_config import GsTrainConfig
from geo_forge.train.model import GaussianSplattingModel


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
    model = GaussianSplattingModel(num_gaussians=config.num_gaussians, device=device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

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

        pred = model.render(intrinsics=intrinsics, c2w=c2w, width=width, height=height)
        loss = F.l1_loss(pred, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_wandb:
            wandb.log({"loss": loss.item()}, step=step + 1)

        if (step + 1) % config.log_interval == 0:
            scene = sample["scene"]
            camera = sample["camera"]
            timestamp = sample["timestamp"]
            print(
                f"[step {step + 1:04d}] "
                f"loss={loss.item():.4f} scene={scene} cam={camera} ts={timestamp}"
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

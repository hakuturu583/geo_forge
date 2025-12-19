from __future__ import annotations

from pathlib import Path

from geo_forge.train.train import (
    LossWeightConfig,
    SequenceTrainConfig,
    build_arg_parser,
    train_sequence,
)


def parse_args() -> SequenceTrainConfig:
    parser = build_arg_parser(
        "Train a single front-camera sequence with SHARP-initialized Gaussians."
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

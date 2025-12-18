from __future__ import annotations

import argparse
import os
from typing import Sequence

from geo_forge.dataset import GeoForgeDataset


def _require_single(value: str | None, *, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} is required.")
    return value.strip()


def train(
    *,
    scene: str,
    camera: str,
    nuscenes_version: str = os.getenv("NUSCENES_VERSION", "v1.0-mini"),
) -> tuple[GeoForgeDataset, GeoForgeDataset]:
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

    Returns:
        (sample_dataset, sweep_dataset)
    """
    scene_name = _require_single(scene, name="scene")
    camera_name = _require_single(camera, name="camera")
    scene_filter: Sequence[str] = [scene_name]
    camera_filter: Sequence[str] = [camera_name]

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

    return sample_dataset, sweep_dataset


def train_one_camera_sequence(
    *,
    scene: str,
    camera: str,
    nuscenes_version: str = os.getenv("NUSCENES_VERSION", "v1.0-mini"),
) -> tuple[GeoForgeDataset, GeoForgeDataset]:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dataset, sweep_dataset = train(
        scene=args.scene,
        camera=args.camera,
        nuscenes_version=args.nuscenes_version,
    )
    print(
        "Prepared datasets:",
        f"sample_only={len(sample_dataset)}",
        f"sample_plus_sweeps={len(sweep_dataset)}",
    )


if __name__ == "__main__":
    main()

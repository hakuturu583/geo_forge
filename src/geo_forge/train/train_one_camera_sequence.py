from __future__ import annotations

import argparse
import os
from collections import deque
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Sequence, TypeVar

from sharp.utils.gaussians import Gaussians3D

from geo_forge.dataset import GeoForgeDataset
from geo_forge.preprocess.sharp_util import _concat_gaussians, _load_sharp_gaussians_world

T = TypeVar("T")


def adjacent(iterable: Iterable[T], n: int = 2) -> Iterator[tuple[T, ...]]:
    """
    Yield overlapping windows of size ``n`` from ``iterable``.

    This is similar to C++'s ``std::views::adjacent`` (or Python's
    ``itertools.pairwise`` when ``n == 2``).
    """
    if n <= 0:
        raise ValueError("n must be >= 1.")

    iterator = iter(iterable)
    window: deque[T] = deque(islice(iterator, n), maxlen=n)
    if len(window) < n:
        return

    yield tuple(window)
    for item in iterator:
        window.append(item)
        yield tuple(window)


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

    # Iterate adjacent sample frames (C++ std::views::adjacent-like).
    # This is a training-loop skeleton; actual computation can be added later.
    sample_metas = sorted(
        sample_dataset.samples,
        key=lambda sample: int(sample["timestamp"]),
    )
    for prev_meta, curr_meta in adjacent(sample_metas, n=2):
        merged_gaussians = _merge_adjacent_sharp_gaussians(prev_meta, curr_meta)
        if merged_gaussians is None:
            continue
        print(
            "Merged SHARP Gaussians:",
            f"prev_ts={int(prev_meta['timestamp'])}",
            f"curr_ts={int(curr_meta['timestamp'])}",
            f"count={int(merged_gaussians.mean_vectors.shape[0])}",
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

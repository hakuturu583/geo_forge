from __future__ import annotations

from geo_forge.dataset import GeoForgeDataset
from geo_forge.preprocess.sharp_util import (
    _concat_gaussians,
    _load_sharp_gaussians_world,
    filter_gaussians_by_distance,
)
from sharp.utils.gaussians import Gaussians3D


def merge_sharp_gaussians_by_distance(dataset: GeoForgeDataset) -> Gaussians3D:
    """
    Merge SHARP Gaussians by keeping only those closest to each sample's pose.
    """
    camera_pose_kdtree = dataset.build_camera_pose_kdtree()
    filtered_gaussians: list[Gaussians3D] = []

    for sample_index, sample in enumerate(dataset.samples):
        gaussians = _load_sharp_gaussians_world(sample)
        if gaussians is None:
            continue
        filtered = filter_gaussians_by_distance(
            camera_pose_kdtree, gaussians, sample_index
        )
        if filtered.mean_vectors.numel() == 0:
            continue
        filtered_gaussians.append(filtered)

    if not filtered_gaussians:
        raise ValueError("No Gaussians3D available to merge after filtering.")

    return _concat_gaussians(filtered_gaussians)


__all__ = ["merge_sharp_gaussians_by_distance"]

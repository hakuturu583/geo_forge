from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from geo_forge.dataset import GeoForgeDataset
from geo_forge.preprocess.sharp_util import (
    _concat_gaussians,
    _ensure_batch_gaussians,
    _load_sharp_gaussians_world,
    filter_gaussians_by_distance,
)
from sharp.utils import color_space as cs_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    PlyData,
    PlyElement,
    convert_rgb_to_spherical_harmonics,
)


def merge_sharp_gaussians_by_distance(dataset: GeoForgeDataset) -> Gaussians3D:
    """
    Merge SHARP Gaussians by keeping only those closest to each sample's pose.
    """
    print("Building camera pose KD-tree...")
    camera_pose_kdtree = dataset.build_camera_pose_kdtree()
    filtered_gaussians: list[Gaussians3D] = []
    processed = 0
    skipped = 0

    for sample_index, sample in tqdm(
        enumerate(dataset.samples),
        total=len(dataset.samples),
        desc="Processing samples",
        unit="sample",
    ):
        # Camera frame AABB; flip Z bounds if your SHARP camera forward is -Z.
        bbox_min = (-15.0, -15.0, 0.0)
        bbox_max = (15.0, 15.0, 30.0)
        gaussians = _load_sharp_gaussians_world(
            sample,
            bounding_box_m=(bbox_min, bbox_max),
        )
        if gaussians is None:
            skipped += 1
            continue
        filtered = filter_gaussians_by_distance(
            camera_pose_kdtree, gaussians, sample_index
        )
        if filtered.mean_vectors.numel() == 0:
            skipped += 1
            continue
        filtered_gaussians.append(filtered)
        processed += 1

    if not filtered_gaussians:
        raise ValueError("No Gaussians3D available to merge after filtering.")

    print(
        "Finished filtering: "
        f"kept={processed} skipped={skipped} total={len(dataset.samples)}"
    )
    return _concat_gaussians(filtered_gaussians)


def _save_ply_sampled_quantile(
    gaussians: Gaussians3D,
    f_px: float,
    image_shape: tuple[int, int],
    path: Path,
    *,
    max_samples: int = 1_000_000,
) -> None:
    """Save Gaussians3D to PLY with sampled disparity quantiles for huge inputs."""

    def _inverse_sigmoid(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor / (1.0 - tensor))

    xyz = gaussians.mean_vectors.flatten(0, 1)
    scale_logits = torch.log(gaussians.singular_values).flatten(0, 1)
    quaternions = gaussians.quaternions.flatten(0, 1)

    colors = convert_rgb_to_spherical_harmonics(
        cs_utils.linearRGB2sRGB(gaussians.colors.flatten(0, 1))
    )
    color_space_index = cs_utils.encode_color_space("sRGB")
    opacity_logits = _inverse_sigmoid(gaussians.opacities).flatten(0, 1).unsqueeze(-1)

    attributes = torch.cat(
        (
            xyz,
            colors,
            opacity_logits,
            scale_logits,
            quaternions,
        ),
        dim=1,
    )

    dtype_full = [
        (attribute, "f4")
        for attribute in ["x", "y", "z"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    ]

    num_gaussians = len(xyz)
    elements = np.empty(num_gaussians, dtype=dtype_full)
    elements[:] = list(map(tuple, attributes.detach().cpu().numpy()))
    vertex_elements = PlyElement.describe(elements, "vertex")

    image_height, image_width = image_shape

    dtype_image_size = [("image_size", "u4")]
    image_size_array = np.empty(2, dtype=dtype_image_size)
    image_size_array[:] = np.array([image_width, image_height])
    image_size_element = PlyElement.describe(image_size_array, "image_size")

    dtype_intrinsic = [("intrinsic", "f4")]
    intrinsic_array = np.empty(9, dtype=dtype_intrinsic)
    intrinsic = np.array(
        [
            f_px,
            0,
            image_width * 0.5,
            0,
            f_px,
            image_height * 0.5,
            0,
            0,
            1,
        ]
    )
    intrinsic_array[:] = intrinsic.flatten()
    intrinsic_element = PlyElement.describe(intrinsic_array, "intrinsic")

    dtype_extrinsic = [("extrinsic", "f4")]
    extrinsic_array = np.empty(16, dtype=dtype_extrinsic)
    extrinsic_array[:] = np.eye(4).flatten()
    extrinsic_element = PlyElement.describe(extrinsic_array, "extrinsic")

    dtype_frames = [("frame", "i4")]
    frame_array = np.empty(2, dtype=dtype_frames)
    frame_array[:] = np.array([1, num_gaussians], dtype=np.int32)
    frame_element = PlyElement.describe(frame_array, "frame")

    dtype_disparity = [("disparity", "f4")]
    disparity_array = np.empty(2, dtype=dtype_disparity)
    disparity = 1.0 / gaussians.mean_vectors[0, ..., -1]
    disparity_flat = disparity.detach().cpu().numpy().reshape(-1)
    if disparity_flat.size > max_samples:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(disparity_flat.size, size=max_samples, replace=False)
        disparity_sample = disparity_flat[sample_idx]
    else:
        disparity_sample = disparity_flat
    quantiles = np.quantile(disparity_sample, q=[0.1, 0.9]).astype(np.float32)
    disparity_array[:] = quantiles
    disparity_element = PlyElement.describe(disparity_array, "disparity")

    dtype_color_space = [("color_space", "u1")]
    color_space_array = np.empty(1, dtype=dtype_color_space)
    color_space_array[:] = np.array([color_space_index]).flatten()
    color_space_element = PlyElement.describe(color_space_array, "color_space")

    dtype_version = [("version", "u1")]
    version_array = np.empty(3, dtype=dtype_version)
    version_array[:] = np.array([1, 5, 0], dtype=np.uint8).flatten()
    version_element = PlyElement.describe(version_array, "version")

    plydata = PlyData(
        [
            vertex_elements,
            extrinsic_element,
            intrinsic_element,
            image_size_element,
            frame_element,
            disparity_element,
            color_space_element,
            version_element,
        ]
    )
    plydata.write(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge SHARP Gaussians by distance and save as initial_gaussians.ply."
        )
    )
    parser.add_argument("--scene", required=True, help="Scene name to process.")
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated list of camera names (e.g. CAM_FRONT,CAM_BACK).",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="NuScenes version (defaults to NUSCENES_VERSION env var).",
    )
    parser.add_argument(
        "--only-sample-frames",
        action="store_true",
        help="Only use synchronized sample frames instead of all sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cameras = None
    if args.cameras:
        cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]

    print("Initializing GeoForgeDataset...")
    dataset = GeoForgeDataset(
        version=args.version,
        scene_filter=[args.scene],
        camera_filter=cameras,
        only_sample_frames=args.only_sample_frames,
    )
    print(f"Loaded {len(dataset.samples)} samples for scene '{args.scene}'.")

    print("Merging Gaussians...")
    merged = merge_sharp_gaussians_by_distance(dataset)
    print(f"Merged Gaussians count: {merged.mean_vectors.shape[0]}")

    sample0 = dataset[0]
    intrinsics = sample0["intrinsics"]
    f_px = float(intrinsics[0, 0].item())
    height = int(sample0["height"])
    width = int(sample0["width"])

    output_dir = dataset.dataset_root / args.scene
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "initial_gaussians.ply"
    print(f"Saving merged Gaussians to {output_path}...")
    merged_batched = _ensure_batch_gaussians(merged)
    _save_ply_sampled_quantile(merged_batched, f_px, (height, width), output_path)
    print("Done.")


if __name__ == "__main__":
    main()


__all__ = ["merge_sharp_gaussians_by_distance"]

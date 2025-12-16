from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from dotenv import load_dotenv

from geo_forge.nuscenes import iterate_all_sweep_camera_frames

load_dotenv()


def _parse_timestamp(stem: str) -> int:
    """Extract the leading timestamp from a filename stem."""
    token = stem.split("_", 1)[0]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse timestamp from filename stem '{stem}'. "
            "Expected '<timestamp>_<camera>...'"
        ) from exc


def _to_image_tensor(path: Path) -> tuple[torch.Tensor, int, int]:
    """Load an RGB image and return CHW float tensor in [0, 1] with spatial dims."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, width, height


class RoseNuScenesDataset(Dataset[dict[str, object]]):
    """
    Dataset that pairs ROSE object-removed frames with NuScenes camera poses.

    The loader expects the directory layout produced by ``rose_preprocessor``,
    i.e. ``<dataset_root>/<scene>/<camera>/object_removed_images/object_removed_images/*.png``.
    Filenames must start with the NuScenes timestamp used for the original mask.
    """

    def __init__(
        self,
        version: str | None = None,
        scene_filter: Sequence[str] | None = None,
        camera_filter: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        dataset_root_env = os.getenv("GEOFORGE_DATASET_ROOT")
        if not dataset_root_env:
            raise EnvironmentError(
                "GEOFORGE_DATASET_ROOT must be set to the directory containing ROSE outputs."
            )
        self.dataset_root = Path(dataset_root_env).expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"GEOFORGE_DATASET_ROOT path not found at {self.dataset_root}"
            )

        dataroot_env = os.getenv("NUSCENES_DATAROOT")
        if not dataroot_env:
            raise EnvironmentError(
                "NUSCENES_DATAROOT must be set to your NuScenes dataroot."
            )
        self.dataroot = Path(dataroot_env).expanduser()
        if not self.dataroot.exists():
            raise FileNotFoundError(
                f"NuScenes dataroot not found at {self.dataroot}. "
                "Set NUSCENES_DATAROOT to your dataset path."
            )

        nusc_version = version or os.getenv("NUSCENES_VERSION", "v1.0-mini")
        self.nusc = NuScenes(
            version=nusc_version, dataroot=str(self.dataroot), verbose=False
        )
        self.scene_filter = set(scene_filter) if scene_filter else None
        self.camera_filter = (
            {c.lower() for c in camera_filter} if camera_filter else None
        )

        self._pose_index = self._build_pose_index()
        self._ego_positions = self._collect_ego_positions()
        self.samples: list[dict[str, object]] = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                "No ROSE outputs found. Run the preprocessing pipeline first or "
                "adjust scene/camera filters."
            )

    def _build_pose_index(self) -> dict[tuple[str, str, int], Dict[str, object]]:
        """
        Map (scene, camera, timestamp) -> NuScenes sample_data entry, including sweeps.

        We index both camera timestamps and the LiDAR timestamp for keyframes to remain
        compatible with mask stems produced from ``iterate_synchronized_samples``.
        """
        scene_names = sorted(self.scene_filter) if self.scene_filter else None
        camera_names = (
            sorted(c.upper() for c in self.camera_filter)
            if self.camera_filter
            else None
        )

        pose_index: dict[tuple[str, str, int], Dict[str, object]] = {}
        for sample_info in iterate_all_sweep_camera_frames(
            self.nusc, scene_names=scene_names, cameras=camera_names
        ):
            scene_name = sample_info["scene_name"]
            for cam_name, cam_info in sample_info["cameras"].items():
                channel = cam_name.lower()
                timestamp = int(cam_info["timestamp"])
                sample_data = self.nusc.get("sample_data", cam_info["token"])

                pose_index[(scene_name, channel, timestamp)] = sample_data

                if sample_info.get("is_key_frame", True):
                    lidar_ts = int(sample_info["timestamp"])
                    pose_index.setdefault((scene_name, channel, lidar_ts), sample_data)

        return pose_index

    def _candidate_image_dirs(self, cam_dir: Path) -> Iterable[Path]:
        yield cam_dir / "object_removed_images" / "object_removed_images"
        yield cam_dir / "object_removed_images"
        yield cam_dir

    def _collect_ego_positions(self) -> torch.Tensor:
        """
        Gather unique ego pose translations referenced by the indexed sample data.
        """
        translations: list[list[float]] = []
        seen_tokens: set[str] = set()
        for sample_data in self._pose_index.values():
            ego_pose_token = sample_data["ego_pose_token"]
            if ego_pose_token in seen_tokens:
                continue
            seen_tokens.add(ego_pose_token)
            ego_pose = self.nusc.get("ego_pose", ego_pose_token)
            translations.append(ego_pose["translation"])

        if not translations:
            raise RuntimeError("No ego poses found in the indexed NuScenes samples.")

        return torch.tensor(translations, dtype=torch.float32)

    def _collect_samples(self) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for scene_dir in sorted(self.dataset_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            if self.scene_filter and scene_dir.name not in self.scene_filter:
                continue

            for cam_dir in sorted(scene_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue

                cam_name = cam_dir.name.lower()
                if self.camera_filter and cam_name not in self.camera_filter:
                    continue

                image_dir = next(
                    (d for d in self._candidate_image_dirs(cam_dir) if d.exists()), None
                )
                if image_dir is None:
                    continue

                image_paths = sorted(
                    [
                        *image_dir.glob("*.png"),
                        *image_dir.glob("*.jpg"),
                        *image_dir.glob("*.jpeg"),
                    ]
                )
                for image_path in image_paths:
                    timestamp = _parse_timestamp(image_path.stem)
                    pose_meta = self._pose_index.get(
                        (scene_dir.name, cam_name, timestamp)
                    )
                    if pose_meta is None:
                        # Skip silently to keep the sample code lightweight.
                        continue

                    intrinsics, c2w = self._camera_from_sample_data(pose_meta)
                    samples.append(
                        {
                            "image_path": image_path,
                            "intrinsics": intrinsics,
                            "c2w": c2w,
                            "scene": scene_dir.name,
                            "camera": cam_name,
                            "timestamp": timestamp,
                        }
                    )
        return samples

    def _camera_from_sample_data(
        self, sample_data: Dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calibrated = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        cam_to_ego = transform_matrix(
            calibrated["translation"],
            Quaternion(calibrated["rotation"]),
            inverse=False,
        )
        ego_to_world = transform_matrix(
            ego_pose["translation"],
            Quaternion(ego_pose["rotation"]),
            inverse=False,
        )
        cam_to_world = ego_to_world @ cam_to_ego

        intrinsics = torch.tensor(
            np.asarray(calibrated["camera_intrinsic"], dtype=np.float32)
        )
        c2w = torch.tensor(cam_to_world, dtype=torch.float32)
        return intrinsics, c2w

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample = self.samples[idx]
        image_tensor, width, height = _to_image_tensor(sample["image_path"])

        return {
            "image": image_tensor,
            "intrinsics": sample["intrinsics"],
            "c2w": sample["c2w"],
            "width": width,
            "height": height,
            "scene": sample["scene"],
            "camera": sample["camera"],
            "timestamp": sample["timestamp"],
        }

    def get_init_gaussian_means(
        self, num_samples: int, radius: float = 3.0
    ) -> torch.Tensor:
        """
        Sample 3D points within ``radius`` meters of random ego poses.

        Args:
            num_samples: Number of Gaussian mean positions to draw.
            radius: Radius in meters of the sampling circle around each ego pose.

        Returns:
            Tensor of shape (num_samples, 3) with XYZ positions.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if radius <= 0:
            raise ValueError("radius must be positive.")

        centers = self._ego_positions
        num_centers = centers.shape[0]
        if num_centers == 0:
            raise RuntimeError("No ego pose centers available for sampling.")

        device = centers.device
        dtype = centers.dtype

        center_indices = torch.randint(
            low=0, high=num_centers, size=(num_samples,), device=device
        )
        chosen_centers = centers[center_indices]

        radii = torch.sqrt(torch.rand(num_samples, device=device, dtype=dtype))
        radii *= radius
        angles = torch.rand(num_samples, device=device, dtype=dtype) * 2 * torch.pi

        offsets = torch.zeros((num_samples, 3), device=device, dtype=dtype)
        offsets[:, 0] = radii * torch.cos(angles)
        offsets[:, 1] = radii * torch.sin(angles)

        return chosen_centers + offsets

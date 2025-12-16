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
        self.samples: list[dict[str, object]] = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                "No ROSE outputs found. Run the preprocessing pipeline first or "
                "adjust scene/camera filters."
            )

    def _build_pose_index(self) -> dict[tuple[str, str, int], Dict[str, object]]:
        """
        Map (scene, camera, timestamp) -> NuScenes sample_data entry.

        Supports matching by LiDAR timestamp for keyframes to remain compatible with
        mask stems produced from ``iterate_synchronized_samples``.
        """
        scene_name_by_token = {
            scene["token"]: scene["name"] for scene in self.nusc.scene
        }
        sample_token_to_scene: dict[str, str] = {}
        for sample in self.nusc.sample:
            scene_name = scene_name_by_token.get(sample["scene_token"])
            if scene_name is None:
                continue
            if self.scene_filter and scene_name not in self.scene_filter:
                continue
            sample_token_to_scene[sample["token"]] = scene_name

        sample_token_to_lidar_ts: dict[str, int] = {}
        for sample_data in self.nusc.sample_data:
            if not sample_data.get("is_key_frame", True):
                continue
            if sample_data.get("channel") != "LIDAR_TOP":
                continue
            sample_token = sample_data.get("sample_token")
            if sample_token not in sample_token_to_scene:
                continue
            sample_token_to_lidar_ts[sample_token] = int(sample_data["timestamp"])

        pose_index: dict[tuple[str, str, int], Dict[str, object]] = {}
        for sample_data in self.nusc.sample_data:
            channel = sample_data.get("channel", "").lower()
            if not channel.startswith("cam_"):
                continue

            sample_token = sample_data.get("sample_token")
            scene_name = sample_token_to_scene.get(sample_token)
            if scene_name is None:
                continue
            if self.camera_filter and channel not in self.camera_filter:
                continue

            timestamp = int(sample_data["timestamp"])
            pose_index[(scene_name, channel, timestamp)] = sample_data

            if sample_data.get("is_key_frame", True):
                lidar_ts = sample_token_to_lidar_ts.get(sample_token)
                if lidar_ts is not None:
                    pose_index.setdefault((scene_name, channel, lidar_ts), sample_data)

        return pose_index

    def _candidate_image_dirs(self, cam_dir: Path) -> Iterable[Path]:
        yield cam_dir / "object_removed_images" / "object_removed_images"
        yield cam_dir / "object_removed_images"
        yield cam_dir

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
                    image_tensor, width, height = _to_image_tensor(image_path)
                    samples.append(
                        {
                            "image": image_tensor,
                            "intrinsics": intrinsics,
                            "c2w": c2w,
                            "width": width,
                            "height": height,
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
        return self.samples[idx]

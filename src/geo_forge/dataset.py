from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Sequence, TypedDict

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from dotenv import load_dotenv

from geo_forge.nuscenes import (
    iterate_all_sweep_camera_frames,
    iterate_synchronized_samples,
    lidar_depth_from_synchronized_sample,
)

load_dotenv()

_NUSC_CAM_TO_OPENGL = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
_NUSC_WORLD_TO_GS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


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


def _resolve_mask_path(
    mask_dir: Path, timestamp: int, camera: str, layer_name: str
) -> Path | None:
    """Return the first existing mask path for the given timestamp/camera/layer."""
    preferred = mask_dir / f"{timestamp}_{camera}_{layer_name}.pt"
    legacy = mask_dir / f"{timestamp}_{layer_name}.pt"
    candidates = [
        preferred,
        legacy,
        *sorted(mask_dir.glob(f"{timestamp}_*_{layer_name}.pt")),
    ]

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_mask(mask_path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Load a mask tensor from disk and validate its shape."""
    width, height = size
    mask_tensor = torch.load(mask_path, map_location="cpu")
    if mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.any(dim=0)
    elif mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.bool()
    else:
        raise ValueError(
            f"Unsupported mask dimensionality {mask_tensor.dim()} in {mask_path}"
        )

    if mask_tensor.shape != (height, width):
        raise ValueError(
            f"Mask shape {tuple(mask_tensor.shape[::-1])} does not match image size "
            f"{(width, height)} for {mask_path}"
        )
    return mask_tensor


class NuScenesData(TypedDict):
    image: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor
    width: int
    height: int
    scene: str
    camera: str
    timestamp: int
    object_mask: torch.Tensor | None
    sky_mask: torch.Tensor | None
    lidar_depth: torch.Tensor | None
    lidar_valid: torch.Tensor | None
    sharp_predicted_gaussians3d: str | None
    nusc_sample_token: str | None
    nusc_sample_data_token: str | None


class GeoForgeDataset(Dataset[NuScenesData]):
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
        *,
        only_sample_frames: bool = False,
        musiq_bottom_fraction: float = 0.2,
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
        self.only_sample_frames = bool(only_sample_frames)

        self._pose_index = self._build_pose_index()
        self._ego_positions = self._collect_ego_positions()
        self.musiq_bottom_fraction = float(musiq_bottom_fraction)
        self.samples, self.skipped_mask_count = self._collect_samples()
        initial_sample_count = len(self.samples)
        self.musiq_filtered_count = 0
        if self.samples:
            self.samples, self.musiq_filtered_count = self._filter_samples_by_musiq(
                self.samples, self.musiq_bottom_fraction
            )
        if self.samples:
            print(
                "MUSIQ filter dropped "
                f"{self.musiq_filtered_count} / {initial_sample_count} samples; "
                f"final={len(self.samples)}"
            )
        self.samples: list[dict[str, object]]
        if not self.samples:
            raise RuntimeError(
                "No ROSE outputs found. Run the preprocessing pipeline first or "
                "adjust scene/camera filters."
            )

    def _build_pose_index(self) -> dict[tuple[str, str, int], Dict[str, object]]:
        """
        Map (scene, camera, timestamp) -> NuScenes sample_data entry, including sweeps.

        When ``only_sample_frames`` is enabled, the index is built from
        ``iterate_synchronized_samples`` (keyframes where LiDAR and camera are
        synchronized). Otherwise, it uses ``iterate_all_sweep_camera_frames``
        which includes intermediate sweep frames for each camera.

        We index both camera timestamps and the keyframe LiDAR timestamp to remain
        compatible with mask stems produced from ``iterate_synchronized_samples``.
        """
        scene_names = sorted(self.scene_filter) if self.scene_filter else None
        camera_names = (
            sorted(c.upper() for c in self.camera_filter)
            if self.camera_filter
            else None
        )

        pose_index: dict[tuple[str, str, int], Dict[str, object]] = {}
        iterator = (
            iterate_synchronized_samples
            if self.only_sample_frames
            else iterate_all_sweep_camera_frames
        )
        for sample_info in iterator(
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

        centers_nusc = torch.tensor(translations, dtype=torch.float32)
        world_to_gs = torch.tensor(_NUSC_WORLD_TO_GS[:3, :3], dtype=torch.float32)
        return centers_nusc @ world_to_gs.T

    def _collect_samples(self) -> tuple[list[dict[str, object]], int]:
        samples: list[dict[str, object]] = []
        skipped_for_masks = 0
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
                if not image_paths:
                    continue

                mask_dir = cam_dir / "mask"
                if not mask_dir.exists():
                    skipped_for_masks += len(image_paths)
                    continue
                for image_path in image_paths:
                    timestamp = _parse_timestamp(image_path.stem)
                    pose_meta = self._pose_index.get(
                        (scene_dir.name, cam_name, timestamp)
                    )
                    if pose_meta is None:
                        # Skip silently to keep the sample code lightweight.
                        continue

                    object_mask_path = _resolve_mask_path(
                        mask_dir, timestamp, cam_name, "movable_objects"
                    )
                    sky_mask_path = _resolve_mask_path(
                        mask_dir, timestamp, cam_name, "sky"
                    )
                    if object_mask_path is None or sky_mask_path is None:
                        skipped_for_masks += 1
                        continue

                    intrinsics, c2w = self._camera_from_sample_data(pose_meta)
                    sharp_ply = cam_dir / "sharp" / f"{timestamp}_sharp.ply"
                    samples.append(
                        {
                            "image_path": image_path,
                            "intrinsics": intrinsics,
                            "c2w": c2w,
                            "scene": scene_dir.name,
                            "camera": cam_name,
                            "timestamp": timestamp,
                            "object_mask_path": object_mask_path,
                            "sky_mask_path": sky_mask_path,
                            "sharp_predicted_gaussians3d": (
                                str(sharp_ply) if sharp_ply.exists() else None
                            ),
                            "nusc_sample_token": pose_meta["sample_token"],
                            "nusc_sample_data_token": pose_meta["token"],
                        }
                    )
        return samples, skipped_for_masks

    def _filter_samples_by_musiq(
        self, samples: list[dict[str, object]], bottom_fraction: float
    ) -> tuple[list[dict[str, object]], int]:
        if bottom_fraction <= 0.0:
            return samples, 0
        if bottom_fraction >= 1.0:
            raise ValueError("musiq_bottom_fraction must be in [0.0, 1.0).")

        from pyarrow import parquet as pq
        import pyarrow as pa
        from tqdm import tqdm

        def cache_root() -> Path:
            root_override = os.getenv("GEOFORGE_DATASET_DIR")
            return (
                Path(root_override).expanduser() if root_override else self.dataset_root
            )

        def cache_path(scene_name: str) -> Path:
            return cache_root() / scene_name / "musiq_score.paraquet"

        def cache_key(scene_name: str, image_path: Path) -> str:
            scene_dir = self.dataset_root / scene_name
            try:
                return str(image_path.relative_to(scene_dir))
            except ValueError:
                return image_path.name

        caches: dict[str, dict[str, float]] = {}
        cache_paths: dict[str, Path] = {}
        for sample in samples:
            scene_name = str(sample["scene"])
            if scene_name in caches:
                continue
            path = cache_path(scene_name)
            cache_paths[scene_name] = path
            if path.exists():
                table = pq.read_table(path)
                data = table.to_pydict()
                files = data.get("file", [])
                scores = data.get("musiq", [])
                caches[scene_name] = {
                    str(file_name): float(score)
                    for file_name, score in zip(files, scores)
                }
            else:
                caches[scene_name] = {}

        metric = None
        device = None
        new_entries: dict[str, dict[str, float]] = {}

        scores: list[float] = []
        with torch.no_grad():
            for sample in tqdm(samples, desc="MUSIQ scoring", unit="image"):
                scene_name = str(sample["scene"])
                image_path = sample["image_path"]
                key = cache_key(scene_name, image_path)
                cached = caches.get(scene_name, {}).get(key)
                if cached is not None:
                    scores.append(float(cached))
                    continue

                if metric is None:
                    import pyiqa

                    device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                    metric = pyiqa.create_metric("musiq")
                    metric.eval()
                    metric.to(device)

                image_tensor, _, _ = _to_image_tensor(image_path)
                input_tensor = image_tensor.unsqueeze(0).to(device)
                score_tensor = metric(input_tensor)
                score = float(score_tensor.squeeze().detach().cpu())
                scores.append(score)
                new_entries.setdefault(scene_name, {})[key] = score

        for scene_name, updates in new_entries.items():
            path = cache_paths.get(scene_name, cache_path(scene_name))
            cache = caches.get(scene_name, {}).copy()
            cache.update(updates)
            files = list(cache.keys())
            musiq_scores = [cache[file_name] for file_name in files]
            table = pa.table({"file": files, "musiq": musiq_scores})
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)

        num_drop = int(len(samples) * bottom_fraction)
        if num_drop == 0:
            return samples, 0
        ranked_indices = np.argsort(scores)
        drop_indices = set(ranked_indices[:num_drop].tolist())
        filtered = [
            sample for index, sample in enumerate(samples) if index not in drop_indices
        ]
        return filtered, num_drop

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
        cam_to_world_nusc = ego_to_world @ cam_to_ego
        # Keep NuScenes camera frame (x right, y down, z forward) to stay aligned
        # with SHARP Gaussians (OpenCV-style camera coordinates).
        cam_to_world = _NUSC_WORLD_TO_GS @ cam_to_world_nusc

        intrinsics = torch.tensor(
            np.asarray(calibrated["camera_intrinsic"], dtype=np.float32)
        )
        c2w = torch.tensor(cam_to_world, dtype=torch.float32)
        return intrinsics, c2w

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> NuScenesData:
        sample = self.samples[idx]
        image_tensor, width, height = _to_image_tensor(sample["image_path"])

        object_mask_path: Path | None = sample.get("object_mask_path")
        sky_mask_path: Path | None = sample.get("sky_mask_path")
        object_mask = (
            _load_mask(object_mask_path, (width, height))
            if object_mask_path is not None
            else None
        )
        sky_mask = (
            _load_mask(sky_mask_path, (width, height))
            if sky_mask_path is not None
            else None
        )

        lidar_depth, lidar_valid = self._load_lidar_depth(
            sample=sample, width=width, height=height
        )

        return {
            "image": image_tensor,
            "intrinsics": sample["intrinsics"],
            "c2w": sample["c2w"],
            "width": width,
            "height": height,
            "scene": sample["scene"],
            "camera": sample["camera"],
            "timestamp": sample["timestamp"],
            "object_mask": object_mask,
            "sky_mask": sky_mask,
            "lidar_depth": lidar_depth,
            "lidar_valid": lidar_valid,
            "sharp_predicted_gaussians3d": sample.get("sharp_predicted_gaussians3d"),
            "nusc_sample_token": sample.get("nusc_sample_token"),
            "nusc_sample_data_token": sample.get("nusc_sample_data_token"),
        }

    def _load_lidar_depth(
        self, *, sample: dict[str, object], width: int, height: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        sample_token = sample.get("nusc_sample_token")
        sample_data_token = sample.get("nusc_sample_data_token")
        if sample_token is None or sample_data_token is None:
            return None, None

        camera = str(sample["camera"]).upper()
        sample_record = self.nusc.get("sample", sample_token)
        lidar_token = sample_record.get("data", {}).get("LIDAR_TOP")
        if lidar_token is None:
            return None, None

        lidar_data = self.nusc.get("sample_data", lidar_token)
        cam_data = self.nusc.get("sample_data", sample_data_token)
        sample_info = {
            "sample_token": sample_token,
            "scene_name": sample.get("scene"),
            "timestamp": lidar_data.get("timestamp"),
            "lidar": {
                "token": lidar_token,
                "filename": lidar_data["filename"],
                "timestamp": lidar_data["timestamp"],
                "calibrated_sensor_token": lidar_data["calibrated_sensor_token"],
                "ego_pose_token": lidar_data["ego_pose_token"],
            },
            "cameras": {
                camera: {
                    "token": sample_data_token,
                    "filename": cam_data["filename"],
                    "timestamp": cam_data["timestamp"],
                    "calibrated_sensor_token": cam_data["calibrated_sensor_token"],
                    "ego_pose_token": cam_data["ego_pose_token"],
                }
            },
            "is_key_frame": True,
        }

        depth_np, mask_np, _ = lidar_depth_from_synchronized_sample(
            self.nusc,
            sample_info,
            camera,
            dataroot=self.dataroot,
            image_size=(width, height),
        )
        depth = torch.from_numpy(depth_np)
        valid = torch.from_numpy(mask_np)
        return depth, valid

    def get_samples_between(
        self, start_timestamp: int, end_timestamp: int, *, inclusive: bool = True
    ) -> list[NuScenesData]:
        """
        Return samples whose NuScenes timestamps fall within the given range.

        Args:
            start_timestamp: Start timestamp (typically NuScenes microseconds).
            end_timestamp: End timestamp (typically NuScenes microseconds).
            inclusive: When True, includes endpoints (start<=t<=end). When False,
                uses an open interval (start<t<end).

        Returns:
            List of samples in the same format as ``__getitem__``.
        """
        if start_timestamp > end_timestamp:
            raise ValueError("start_timestamp must be <= end_timestamp.")

        if inclusive:
            indices = [
                idx
                for idx, sample in enumerate(self.samples)
                if start_timestamp <= int(sample["timestamp"]) <= end_timestamp
            ]
        else:
            indices = [
                idx
                for idx, sample in enumerate(self.samples)
                if start_timestamp < int(sample["timestamp"]) < end_timestamp
            ]

        return [self.__getitem__(idx) for idx in indices]

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

        # centers are already stored in the gsplat world frame; keep offsets in the
        # same frame and avoid reapplying the NuScenes->gsplat conversion.
        return chosen_centers + offsets

    def build_camera_pose_kdtree(
        self,
    ) -> tuple[cKDTree, dict[tuple[float, float, float], list[int]]]:
        """
        Build a 3D KD-tree from all camera-to-world translations.

        Returns:
            Tuple of (KD-tree, seed-to-sample-index mapping) built from unique
            camera positions.
        """
        translations: list[torch.Tensor] = []
        seed_to_samples: dict[tuple[float, float, float], list[int]] = {}
        for sample_index, sample in enumerate(self.samples):
            translation = (
                sample["c2w"][:3, 3].detach().to(dtype=torch.float32, device="cpu")
            )
            translations.append(translation)

            key = tuple(float(value) for value in translation.tolist())
            seed_to_samples.setdefault(key, []).append(sample_index)
        if not translations:
            raise RuntimeError("No camera poses available to build Voronoi diagram.")

        points = torch.stack(translations, dim=0).numpy()
        points = np.unique(points, axis=0)
        return cKDTree(points), seed_to_samples

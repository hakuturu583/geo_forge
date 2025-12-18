"""Preprocessing functions for NuScenes dataset"""

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from PIL import Image as PilImage
from pyquaternion import Quaternion
from dotenv import load_dotenv

from geo_forge.dataclass import NuscenesObjectBoundingBox

# Load environment variables
load_dotenv()


def iterate_synchronized_samples(
    nusc: NuScenes,
    scene_names: Optional[List[str]] = None,
    cameras: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate through NuScenes frames with synchronized image and LiDAR timestamps.

    Only keyframe samples are yielded. Use ``iterate_all_sweep_camera_frames`` to
    include intermediate sweep frames for each camera.

    Args:
        nusc: NuScenes instance
        scene_names: List of scene names to process (None for all scenes)
        cameras: List of cameras to use (default is all 6 cameras)

    Yields:
        Dictionary containing sample information:
            - sample_token: Sample token
            - timestamp: Timestamp (microseconds)
            - lidar: LiDAR data information
            - cameras: Camera data information for each camera
            - annotations: List[ObjectBoundingBox] for 3D bounding boxes
            - is_key_frame: Whether the frame is a keyframe sample or sweep
    """
    # Default camera list
    if cameras is None:
        cameras = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]

    # Get scenes to process
    scenes = nusc.scene
    if scene_names is not None:
        scenes = [s for s in scenes if s["name"] in scene_names]

    for scene in scenes:
        scene_name = scene["name"]
        # Start from the first sample in the scene
        sample_token = scene["first_sample_token"]

        while sample_token != "":
            sample = nusc.get("sample", sample_token)

            # Get LiDAR data
            lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = nusc.get("sample_data", lidar_token)
            lidar_timestamp = lidar_data["timestamp"]

            # Collect synchronized camera data
            synchronized_cameras = {}
            all_synchronized = True

            for cam_name in cameras:
                if cam_name not in sample["data"]:
                    continue

                cam_token = sample["data"][cam_name]
                cam_data = nusc.get("sample_data", cam_token)

                # Check if timestamps match
                # In NuScenes, sensor data within a keyframe (sample) is
                # basically synchronized, so data within the same sample
                # has nearly identical timestamps
                if abs(cam_data["timestamp"] - lidar_timestamp) < 50000:  # Within 50ms
                    synchronized_cameras[cam_name] = {
                        "token": cam_token,
                        "filename": cam_data["filename"],
                        "timestamp": cam_data["timestamp"],
                        "calibrated_sensor_token": cam_data["calibrated_sensor_token"],
                        "ego_pose_token": cam_data["ego_pose_token"],
                    }
                else:
                    all_synchronized = False
                    break

            # Gather 3D bounding boxes for the sample
            annotations: List[NuscenesObjectBoundingBox] = []
            for ann_token in sample.get("anns", []):
                ann_record = nusc.get("sample_annotation", ann_token)
                annotations.append(
                    NuscenesObjectBoundingBox.from_sample_annotation(ann_record)
                )

            # Yield only if all cameras are synchronized
            if all_synchronized and len(synchronized_cameras) > 0:
                yield {
                    "sample_token": sample_token,
                    "scene_name": scene_name,
                    "timestamp": lidar_timestamp,
                    "lidar": {
                        "token": lidar_token,
                        "filename": lidar_data["filename"],
                        "timestamp": lidar_timestamp,
                        "calibrated_sensor_token": lidar_data[
                            "calibrated_sensor_token"
                        ],
                        "ego_pose_token": lidar_data["ego_pose_token"],
                    },
                    "cameras": synchronized_cameras,
                    "annotations": annotations,
                    "is_key_frame": True,
                }

            # Move to next sample
            sample_token = sample["next"]


def iterate_all_sweep_camera_frames(
    nusc: NuScenes,
    scene_names: Optional[List[str]] = None,
    cameras: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Iterate through keyframe samples and all subsequent camera sweep frames.

    This includes keyframe samples (matching ``iterate_synchronized_samples``) and
    every intermediate non-keyframe camera frame until the next keyframe. LiDAR
    metadata remains the keyframe reading for sweeps.

    Args:
        nusc: NuScenes instance
        scene_names: List of scene names to process (None for all scenes)
        cameras: List of cameras to use (default is all 6 cameras)

    Yields:
        Dictionary with the same structure as ``iterate_synchronized_samples``,
        marking sweeps with ``is_key_frame=False``.
    """
    for sample_info in iterate_synchronized_samples(
        nusc, scene_names=scene_names, cameras=cameras
    ):
        yield sample_info

        lidar_entry = sample_info["lidar"]
        annotations = sample_info.get("annotations", [])
        for cam_name, cam_info in sample_info["cameras"].items():
            sweep_token = nusc.get("sample_data", cam_info["token"]).get("next")
            while sweep_token:
                sweep_data = nusc.get("sample_data", sweep_token)
                if sweep_data["is_key_frame"]:
                    break

                yield {
                    "sample_token": sample_info["sample_token"],
                    "scene_name": sample_info["scene_name"],
                    "timestamp": sweep_data["timestamp"],
                    "lidar": lidar_entry,
                    "cameras": {
                        cam_name: {
                            "token": sweep_token,
                            "filename": sweep_data["filename"],
                            "timestamp": sweep_data["timestamp"],
                            "calibrated_sensor_token": sweep_data[
                                "calibrated_sensor_token"
                            ],
                            "ego_pose_token": sweep_data["ego_pose_token"],
                        }
                    },
                    "annotations": annotations,
                    "is_key_frame": False,
                }
                sweep_token = sweep_data.get("next")


def load_synchronized_data(
    nusc: NuScenes, sample_info: Dict[str, Any], dataroot: Optional[Path] = None
) -> Tuple[
    Optional[LidarPointCloud],
    Dict[str, Dict[str, Any]],
    List[NuscenesObjectBoundingBox],
]:
    """
    Load LiDAR and image data from synchronized samples

    Args:
        nusc: NuScenes instance
        sample_info: Sample information from iterate_synchronized_samples
        dataroot: Data root path (if None, gets path from nusc)

    Returns:
        (LiDAR point cloud or None for sweep frames, dictionary of camera images as
        Pillow Image, 3D boxes)
    """
    if dataroot is None:
        dataroot = Path(nusc.dataroot)

    # Load LiDAR data only for keyframes; sweeps are not annotated.
    pc: Optional[LidarPointCloud] = None
    lidar_info = sample_info.get("lidar")
    if lidar_info and sample_info.get("is_key_frame", True):
        lidar_path = dataroot / lidar_info["filename"]
        pc = LidarPointCloud.from_file(str(lidar_path))

    # Load camera images
    images = {}
    for cam_name, cam_info in sample_info["cameras"].items():
        img_path = dataroot / cam_info["filename"]
        try:
            with PilImage.open(img_path) as img:
                rgb_image = img.convert("RGB")
        except FileNotFoundError:
            continue

        images[cam_name] = {
            "image": rgb_image,
            "token": cam_info["token"],
            "calibrated_sensor_token": cam_info["calibrated_sensor_token"],
            "ego_pose_token": cam_info["ego_pose_token"],
        }

    return pc, images, sample_info.get("annotations", [])


def project_points_to_depth_image(
    points_cam: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    *,
    min_depth: float = 0.1,
    max_depth: float | None = None,
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame 3D points into an image and build a z-buffer depth map.

    Args:
        points_cam: Camera-frame 3D points with shape (3, N) or (N, 3).
        intrinsics: Camera intrinsics matrix with shape (3, 3).
        width: Image width in pixels.
        height: Image height in pixels.
        min_depth: Minimum accepted depth.
        max_depth: Optional maximum accepted depth.
        fill_value: Depth value for pixels with no projected points (defaults to NaN).

    Returns:
        (depth, mask) where:
          - depth is a float32 array of shape (H, W)
          - mask is a bool array of shape (H, W) indicating valid depth pixels
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive; got {(width, height)}")
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"intrinsics must have shape (3, 3); got {tuple(intrinsics.shape)}"
        )

    points = np.asarray(points_cam, dtype=np.float32)
    if points.ndim != 2:
        raise ValueError(f"points_cam must be 2D; got shape {tuple(points.shape)}")
    if points.shape[0] == 3:
        xyz = points
    elif points.shape[1] == 3:
        xyz = points.T
    else:
        raise ValueError(
            f"points_cam must be shaped (3, N) or (N, 3); got {tuple(points.shape)}"
        )

    depths = xyz[2, :]
    valid = depths > float(min_depth)
    if max_depth is not None:
        valid &= depths < float(max_depth)

    if not np.any(valid):
        depth_out = np.full((height, width), fill_value, dtype=np.float32)
        mask = np.zeros((height, width), dtype=bool)
        return depth_out, mask

    xyz = xyz[:, valid]
    depths = depths[valid]

    uvw = view_points(xyz, intrinsics, normalize=True)
    xs = np.round(uvw[0, :]).astype(np.int32)
    ys = np.round(uvw[1, :]).astype(np.int32)
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[inside]
    ys = ys[inside]
    depths = depths[inside].astype(np.float32)

    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    if xs.size:
        np.minimum.at(depth_buffer, (ys, xs), depths)

    mask = np.isfinite(depth_buffer)
    depth_out = np.where(mask, depth_buffer, np.float32(fill_value)).astype(np.float32)
    return depth_out, mask


def lidar_depth_from_synchronized_sample(
    nusc: NuScenes,
    sample_info: Dict[str, Any],
    camera_name: str,
    *,
    dataroot: Optional[Path] = None,
    image_size: tuple[int, int] | None = None,
    min_depth: float = 0.1,
    max_depth: float | None = None,
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project the keyframe LiDAR point cloud into a camera image and produce depth.

    This consumes dictionaries yielded by ``iterate_synchronized_samples`` (or
    ``iterate_all_sweep_camera_frames``), loads the LiDAR point cloud, transforms
    it into the camera coordinate frame, and builds a per-pixel depth map via a
    z-buffer (nearest point wins).

    Returns:
        (depth, mask, intrinsics) where depth is in the same unit as NuScenes
        (meters), mask indicates valid pixels, and intrinsics is the 3x3 camera K.
    """
    if dataroot is None:
        dataroot = Path(nusc.dataroot)

    lidar_info = sample_info.get("lidar")
    if not lidar_info:
        raise ValueError("sample_info is missing 'lidar' entry.")
    camera_info = sample_info.get("cameras", {}).get(camera_name)
    if camera_info is None:
        raise KeyError(f"Camera '{camera_name}' not found in sample_info['cameras'].")

    lidar_path = dataroot / lidar_info["filename"]
    pc = LidarPointCloud.from_file(str(lidar_path))

    lidar_calib = nusc.get("calibrated_sensor", lidar_info["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_info["ego_pose_token"])
    cam_calib = nusc.get("calibrated_sensor", camera_info["calibrated_sensor_token"])
    cam_pose = nusc.get("ego_pose", camera_info["ego_pose_token"])

    # Lidar sensor -> ego (lidar time)
    pc.rotate(Quaternion(lidar_calib["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_calib["translation"], dtype=np.float32))

    # Ego (lidar time) -> global
    pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_pose["translation"], dtype=np.float32))

    # Global -> ego (camera time)
    pc.translate(-np.array(cam_pose["translation"], dtype=np.float32))
    pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)

    # Ego (camera time) -> camera
    pc.translate(-np.array(cam_calib["translation"], dtype=np.float32))
    pc.rotate(Quaternion(cam_calib["rotation"]).rotation_matrix.T)

    intrinsics = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Unexpected camera_intrinsic shape {tuple(intrinsics.shape)} for {camera_name}"
        )

    if image_size is None:
        img_path = dataroot / camera_info["filename"]
        with PilImage.open(img_path) as img:
            width, height = img.size
    else:
        width, height = image_size

    depth, mask = project_points_to_depth_image(
        points_cam=pc.points[:3, :],
        intrinsics=intrinsics,
        width=int(width),
        height=int(height),
        min_depth=min_depth,
        max_depth=max_depth,
        fill_value=fill_value,
    )
    return depth, mask, intrinsics


def example_usage():
    """Usage example"""
    # Load NuScenes with dataroot from environment variable
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

    # Iterate through synchronized samples
    for i, sample_info in enumerate(iterate_synchronized_samples(nusc)):
        print(f"Sample {i}:")
        print(f"  Scene: {sample_info['scene_name']}")
        print(f"  Timestamp: {sample_info['timestamp']}")
        print(f"  Cameras: {list(sample_info['cameras'].keys())}")

        # Load data
        pc, images, _ = load_synchronized_data(nusc, sample_info)
        if pc is not None:
            print(f"  LiDAR points: {pc.points.shape}")
        for cam_name, img_data in images.items():
            width, height = img_data["image"].size
            print(f"  {cam_name} size: {(width, height)}")

        # # Process only first 5 samples
        # if i >= 4:
        #     break


if __name__ == "__main__":
    # Demo run
    example_usage()

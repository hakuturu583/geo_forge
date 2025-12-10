"""Preprocessing functions for NuScenes dataset"""

import os
from typing import Iterator, Tuple, Dict, Any, List, Optional
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image as PilImage
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
    Iterate through sample frames with synchronized image and LiDAR timestamps

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
                    "scene_name": scene["name"],
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
                }

            # Move to next sample
            sample_token = sample["next"]


def load_synchronized_data(
    nusc: NuScenes, sample_info: Dict[str, Any], dataroot: Optional[Path] = None
) -> Tuple[LidarPointCloud, Dict[str, Dict[str, Any]]]:
    """
    Load LiDAR and image data from synchronized samples

    Args:
        nusc: NuScenes instance
        sample_info: Sample information from iterate_synchronized_samples
        dataroot: Data root path (if None, gets path from nusc)

    Returns:
        (LiDAR point cloud, dictionary of camera images as Pillow Image)
    """
    if dataroot is None:
        dataroot = Path(nusc.dataroot)

    # Load LiDAR data
    lidar_path = dataroot / sample_info["lidar"]["filename"]
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

    return pc, images


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
        pc, images = load_synchronized_data(nusc, sample_info)
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

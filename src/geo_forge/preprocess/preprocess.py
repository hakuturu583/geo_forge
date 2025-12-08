"""Preprocess NuScenes dataset with SAM3 using bounding boxes"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from dotenv import load_dotenv

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

from geo_forge.preprocess.sam3_preprocessor import SAM3Preprocessor
from geo_forge.dataclass import SAM3PreprocessorConfig
from geo_forge.nuscenes import iterate_synchronized_samples

# Load environment variables
load_dotenv()


def project_3d_box_to_2d(
    nusc: NuScenes, box: Box, cam_token: str, min_visibility: int = 1
) -> Optional[List[float]]:
    """
    Project a 3D bounding box to 2D image coordinates

    Args:
        nusc: NuScenes instance
        box: 3D bounding box
        cam_token: Camera sample data token
        min_visibility: Minimum number of visible corners (1-8)

    Returns:
        2D bounding box [x1, y1, x2, y2] or None if not visible
    """
    # Get camera data
    cam_data = nusc.get("sample_data", cam_token)

    # Get camera calibration
    cs_rec = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    camera_intrinsic = np.array(cs_rec["camera_intrinsic"])

    # Get ego pose
    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])

    # Transform box to ego vehicle frame
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(np.array(ego_pose["rotation"]))

    # Transform box to camera frame
    box.translate(-np.array(cs_rec["translation"]))
    box.rotate(np.array(cs_rec["rotation"]))

    # Get 3D corners of the box
    corners_3d = box.corners()

    # Project to 2D
    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]

    # Check if corners are in front of the camera
    depths = corners_3d[2, :]
    if np.all(depths <= 0):
        return None

    # Count visible corners
    visible_corners = np.sum(depths > 0)
    if visible_corners < min_visibility:
        return None

    # Get 2D bounding box
    x_coords = corners_2d[0, depths > 0]
    y_coords = corners_2d[1, depths > 0]

    # Check if box is within image bounds (assuming standard resolution)
    if np.any(x_coords < 0) or np.any(y_coords < 0):
        # Still include if partially visible
        pass

    x1, y1 = np.min(x_coords), np.min(y_coords)
    x2, y2 = np.max(x_coords), np.max(y_coords)

    return [float(x1), float(y1), float(x2), float(y2)]


def get_bounding_boxes_for_frame(
    nusc: NuScenes,
    sample_token: str,
    cam_token: str,
    target_categories: Optional[List[str]] = None,
) -> List[List[float]]:
    """
    Get 2D bounding boxes for all objects in a frame

    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        cam_token: Camera sample data token
        target_categories: List of target categories to include (None for all)

    Returns:
        List of 2D bounding boxes [[x1, y1, x2, y2], ...]
    """
    sample = nusc.get("sample", sample_token)
    boxes_2d = []

    # Default target categories
    if target_categories is None:
        target_categories = [
            "vehicle.car",
            "vehicle.truck",
            "vehicle.bus",
            "vehicle.bicycle",
            "vehicle.motorcycle",
            "human.pedestrian.adult",
            "human.pedestrian.child",
        ]

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # Filter by category
        if not any(cat in ann["category_name"] for cat in target_categories):
            continue

        # Create 3D box
        box = Box(
            ann["translation"],
            ann["size"],
            ann["rotation"],
            name=ann["category_name"],
            token=ann["token"],
        )

        # Project to 2D
        box_2d = project_3d_box_to_2d(nusc, box, cam_token)
        if box_2d is not None:
            boxes_2d.append(box_2d)

    return boxes_2d


def process_nuscenes_scene(
    scene_name: str,
    camera: str = "CAM_FRONT",
    max_frames: Optional[int] = None,
    target_categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a NuScenes scene with SAM3 using bounding box prompts

    Args:
        scene_name: Name of the scene to process
        camera: Camera to use
        max_frames: Maximum number of frames to process
        target_categories: Categories to track
        output_dir: Output directory (relative to package)

    Returns:
        Processing results dictionary
    """
    # Setup paths
    package_dir = Path(__file__).parent
    if output_dir is None:
        output_dir = package_dir / "datasets"
    else:
        output_dir = package_dir / output_dir

    # Load NuScenes
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

    print(f"Processing scene: {scene_name} with camera: {camera}")

    # Get frames for the scene
    frames = []
    frame_infos = []
    bounding_boxes_per_frame = {}

    for i, sample_info in enumerate(
        iterate_synchronized_samples(nusc, scene_names=[scene_name], cameras=[camera])
    ):
        if max_frames and i >= max_frames:
            break

        if camera not in sample_info["cameras"]:
            continue

        # Load image
        img_path = Path(dataroot) / sample_info["cameras"][camera]["filename"]
        img = Image.open(img_path).convert("RGB")
        frames.append(img)
        frame_infos.append(sample_info)

        # Get bounding boxes for first frame only (SAM3 will track in subsequent frames)
        if i == 0:
            cam_token = sample_info["cameras"][camera]["token"]
            boxes = get_bounding_boxes_for_frame(
                nusc, sample_info["sample_token"], cam_token, target_categories
            )
            if boxes:
                bounding_boxes_per_frame[0] = {"boxes": boxes}
                print(f"Found {len(boxes)} objects to track in first frame")

    if not frames:
        print(f"No frames found for scene {scene_name}")
        return {}

    print(f"Loaded {len(frames)} frames")

    # Configure SAM3
    config = SAM3PreprocessorConfig(
        model_id="facebook/sam2-hiera-large",  # Use SAM2 until SAM3 is available
        output_dir=str(output_dir / scene_name / camera),
        save_masks=True,
        save_visualizations=True,
        verbose=True,
    )

    # Initialize preprocessor
    preprocessor = SAM3Preprocessor(config=config)

    # Process video with bounding box prompts
    results = preprocessor.process_video(
        frames=frames,
        prompts=bounding_boxes_per_frame,
        identifier=f"{scene_name}_{camera}",
    )

    # Add metadata
    results["scene_name"] = scene_name
    results["camera"] = camera
    results["frame_tokens"] = [f["sample_token"] for f in frame_infos]

    return results


def main():
    """Main function to process NuScenes scenes"""

    # Process first scene from mini dataset
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)

    # Get first scene
    first_scene = nusc.scene[0]["name"]

    # Process scene with bounding boxes
    results = process_nuscenes_scene(
        scene_name=first_scene,
        camera="CAM_FRONT",
        max_frames=10,  # Process first 10 frames for demo
        target_categories=["vehicle.car", "human.pedestrian.adult"],
        output_dir="datasets",  # Relative to package
    )

    print(f"\nProcessing complete!")
    print(f"Scene: {results.get('scene_name', 'N/A')}")
    print(f"Frames processed: {results.get('num_frames', 0)}")
    print(f"Masks saved to: {results.get('output_dir', 'N/A')}")
    print(f"Number of mask files: {len(results.get('mask_files', []))}")


if __name__ == "__main__":
    main()

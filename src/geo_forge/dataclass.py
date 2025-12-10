"""Data classes for SAM3D preprocessing configuration"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING, Sequence

import torch
from PIL import Image
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points

if TYPE_CHECKING:
    from nuscenes.nuscenes import NuScenes


@dataclass
class ObjectMask:
    """
    Container for segmentation mask predictions produced by SAM3.

    This type wraps the dictionaries returned by
    ``Sam3Processor.post_process_instance_segmentation`` so downstream code can
    operate on a structured object instead of raw dicts.
    """

    masks: torch.Tensor
    scores: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    boxes: Optional[torch.Tensor] = None

    def __init__(
        self,
        masks: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ):
        self.masks = masks
        self.scores = scores
        self.labels = labels
        self.boxes = boxes

    @classmethod
    def from_result_list(
        cls, results: List[Dict[str, torch.Tensor]]
    ) -> List["ObjectMask"]:
        """
        Construct ObjectMask instances from SAM3 post-processed output.

        Args:
            results: Output from ``post_process_instance_segmentation`` (list per
                image containing mask tensors and metadata).

        Returns:
            List of ObjectMask instances mirroring the input list order.
        """
        if not results:
            raise ValueError("results is empty; expected at least one element")

        object_masks: List[ObjectMask] = []
        for result in results:
            object_masks.append(
                cls(
                    masks=result.get("masks", torch.empty(0)),
                    scores=result.get("scores"),
                    labels=result.get("labels"),
                    boxes=result.get("boxes"),
                )
            )
        return object_masks

    def overray_mask(self, image: Image.Image) -> Image.Image:
        """
        Apply the union of masks to the image and black out masked pixels.

        Args:
            image: PIL Image to apply the mask to. Must match mask spatial dimensions.

        Returns:
            New PIL Image with masked regions filled with black.
        """
        if self.masks.numel() == 0:
            return image.copy()

        mask = self.masks.detach().to("cpu")
        if mask.dim() == 3:
            combined_mask = mask.sum(dim=0) > 0
        elif mask.dim() == 2:
            combined_mask = mask > 0
        else:
            raise ValueError(f"Unsupported mask dimensionality: {mask.dim()}")

        combined_mask = combined_mask.cpu().numpy().astype(bool)
        img_rgb = image.convert("RGB")
        if img_rgb.size != (combined_mask.shape[1], combined_mask.shape[0]):
            raise ValueError(
                "Mask and image spatial dimensions do not match: "
                f"mask={combined_mask.shape[::-1]}, image={img_rgb.size}"
            )

        img_arr = np.array(img_rgb)
        img_arr[combined_mask] = 0
        return Image.fromarray(img_arr)


@dataclass
class NuscenesObjectBoundingBox:
    """3D bounding box with metadata sourced from NuScenes annotations."""

    token: str
    translation: Tuple[float, float, float]  # x, y, z in global frame (meters)
    size: Tuple[float, float, float]  # width, length, height (meters)
    rotation: Tuple[float, float, float, float]  # quaternion (w, x, y, z)
    velocity: Optional[Tuple[float, float]] = None  # vx, vy in global frame
    category_name: Optional[str] = None
    instance_token: Optional[str] = None
    num_lidar_pts: Optional[int] = None
    num_radar_pts: Optional[int] = None

    def to_2d_bbox(
        self,
        nusc: "NuScenes",
        calibrated_sensor_token: str,
        ego_pose_token: str,
        image_size: Tuple[int, int],
        ignore_category: Optional[Sequence[str]] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Project the 3D bounding box into the camera image plane.

        Args:
            nusc: NuScenes instance used to resolve calibration and ego pose.
            calibrated_sensor_token: Camera calibrated_sensor token for intrinsics/extrinsics.
            ego_pose_token: Ego pose token for the camera frame at capture time.
            image_size: Image (width, height) used for clipping the projected box.
            ignore_category: Iterable of category names to skip; returns None if matched.

        Returns:
            (xmin, ymin, xmax, ymax) in pixel coordinates if visible, otherwise None.
        """
        if ignore_category and self.category_name in ignore_category:
            return None

        cam_cs = nusc.get("calibrated_sensor", calibrated_sensor_token)
        ego_pose = nusc.get("ego_pose", ego_pose_token)

        box = Box(
            center=np.array(self.translation),
            size=np.array(self.size),
            orientation=Quaternion(self.rotation),
        )

        box.translate(-np.array(ego_pose["translation"]))
        box.rotate(Quaternion(ego_pose["rotation"]).inverse)
        box.translate(-np.array(cam_cs["translation"]))
        box.rotate(Quaternion(cam_cs["rotation"]).inverse)

        corners_cam = box.corners()
        if (corners_cam[2, :] <= 0).any():
            return None

        intrinsic = np.array(cam_cs["camera_intrinsic"])
        corners_2d = view_points(corners_cam, intrinsic, normalize=True)
        x_min, y_min = corners_2d[:2, :].min(axis=1)
        x_max, y_max = corners_2d[:2, :].max(axis=1)

        width, height = image_size
        if x_max <= 0 or y_max <= 0 or x_min >= width or y_min >= height:
            return None

        clipped_x_min = float(np.clip(x_min, 0, width))
        clipped_y_min = float(np.clip(y_min, 0, height))
        clipped_x_max = float(np.clip(x_max, 0, width))
        clipped_y_max = float(np.clip(y_max, 0, height))

        if clipped_x_min >= clipped_x_max or clipped_y_min >= clipped_y_max:
            return None

        return (clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max)

    @classmethod
    def from_sample_annotation(
        cls, ann: Dict[str, object]
    ) -> "NuscenesObjectBoundingBox":
        """Create a NuscenesObjectBoundingBox from a NuScenes sample_annotation record."""
        velocity = ann.get("velocity")  # may be provided by NuScenes helper
        velocity_tuple = None
        if (
            velocity is not None
            and isinstance(velocity, (list, tuple))
            and len(velocity) >= 2
        ):
            velocity_tuple = (float(velocity[0]), float(velocity[1]))

        return cls(
            token=str(ann["token"]),
            translation=tuple(float(x) for x in ann["translation"]),  # type: ignore[arg-type]
            size=tuple(float(x) for x in ann["size"]),  # type: ignore[arg-type]
            rotation=tuple(float(x) for x in ann["rotation"]),  # type: ignore[arg-type]
            velocity=velocity_tuple,
            category_name=str(ann.get("category_name"))
            if ann.get("category_name")
            else None,
            instance_token=str(ann.get("instance_token"))
            if ann.get("instance_token")
            else None,
            num_lidar_pts=int(ann.get("num_lidar_pts"))
            if ann.get("num_lidar_pts") is not None
            else None,
            num_radar_pts=int(ann.get("num_radar_pts"))
            if ann.get("num_radar_pts") is not None
            else None,
        )


def overray_mask(image: Image.Image, masks: List[ObjectMask]) -> Image.Image:
    """
    Apply multiple ObjectMask instances to an image and black out their union.

    Args:
        image: PIL Image to apply the masks to.
        masks: List of ObjectMask predictions.

    Returns:
        New PIL Image with all masked regions filled with black.
    """
    valid_masks: List[torch.Tensor] = []
    for mask_obj in masks:
        if mask_obj.masks.numel() == 0:
            continue

        mask_tensor = mask_obj.masks.detach().to("cpu")
        if mask_tensor.dim() == 3:
            combined_mask = mask_tensor.sum(dim=0) > 0
        elif mask_tensor.dim() == 2:
            combined_mask = mask_tensor > 0
        else:
            raise ValueError(f"Unsupported mask dimensionality: {mask_tensor.dim()}")
        valid_masks.append(combined_mask)

    if not valid_masks:
        return image.copy()

    union_mask = torch.stack(valid_masks).any(dim=0).cpu().numpy().astype(bool)
    img_rgb = image.convert("RGB")
    if img_rgb.size != (union_mask.shape[1], union_mask.shape[0]):
        raise ValueError(
            "Mask and image spatial dimensions do not match: "
            f"mask={union_mask.shape[::-1]}, image={img_rgb.size}"
        )

    img_arr = np.array(img_rgb)
    img_arr[union_mask] = 0
    return Image.fromarray(img_arr)

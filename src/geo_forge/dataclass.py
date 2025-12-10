"""Data classes for SAM3D preprocessing configuration"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from PIL import Image
import numpy as np


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

        mask = self.masks
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

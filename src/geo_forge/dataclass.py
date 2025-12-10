"""Data classes for SAM3D preprocessing configuration"""

from dataclasses import dataclass
from typing import List, Optional, Dict

import torch


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
    def from_result_list(cls, results: List[Dict[str, torch.Tensor]]) -> "ObjectMask":
        """
        Construct an ObjectMask from the SAM3 post-processed output.

        Args:
            results: Output from ``post_process_instance_segmentation`` (list per
                image containing mask tensors and metadata).

        Returns:
            ObjectMask containing the mask tensor and optional metadata.
        """
        if not results:
            raise ValueError("results is empty; expected at least one element")

        first = results[0]
        return cls(
            masks=first.get("masks", torch.empty(0)),
            scores=first.get("scores"),
            labels=first.get("labels"),
            boxes=first.get("boxes"),
        )

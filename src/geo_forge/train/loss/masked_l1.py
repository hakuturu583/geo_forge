from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.config import MaskLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class MaskedL1Loss(LossBase):
    """
    L1 loss with sky and movable-object masks applied.
    """

    name = "masked_l1"

    def __init__(self, config: MaskLossWeightConfig) -> None:
        super().__init__()
        self._config = config

    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        if sample is None:
            raise ValueError("sample is required for masked L1 loss.")
        height = pred.shape[-2]
        width = pred.shape[-1]
        weights = self._build_loss_weights(sample, pred.device, height, width)
        loss_map = F.l1_loss(pred, target, reduction="none")
        weights = weights.expand_as(loss_map).to(loss_map.dtype)
        weight_sum = weights.sum()
        if weight_sum.item() == 0:
            return loss_map.new_tensor(0.0)
        return (loss_map * weights).sum() / weight_sum

    def _build_loss_weights(
        self,
        sample: dict[str, object],
        device: torch.device,
        height: int,
        width: int,
    ) -> torch.Tensor:
        sky_mask = sample.get("sky_mask")
        object_mask = sample.get("object_mask")

        weights = torch.ones((1, height, width), device=device)
        if sky_mask is not None:
            weights = torch.where(
                sky_mask.to(device).unsqueeze(0).bool(),
                torch.tensor(self._config.sky, device=device),
                weights,
            )
        else:
            raise RuntimeError(
                "sky_mask is required but not provided in the sweep sample. "
                "Please run SAM3 preprocessor and generate the masks."
            )

        if object_mask is not None:
            obj_mask = object_mask.to(device).unsqueeze(0)
            weights = torch.where(
                obj_mask.bool(),
                torch.tensor(self._config.movable_objects, device=device),
                weights,
            )
        else:
            raise RuntimeError(
                "object_mask is required but not provided in the sweep sample. "
                "Please run SAM3 preprocessor and generate the masks."
            )
        return weights

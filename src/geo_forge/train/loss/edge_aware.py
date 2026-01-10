from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.config import EdgeAwareLossWeightConfig, MaskLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class EdgeAwareLoss(LossBase):
    """
    Compare images using gradient-domain L1 to emphasize edge alignment.
    """

    name = "edge_aware"

    def __init__(
        self, edge_config: EdgeAwareLossWeightConfig, mask_config: MaskLossWeightConfig
    ) -> None:
        super().__init__()
        self._edge_config = edge_config
        self._mask_config = mask_config

    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must share the same shape; "
                f"got {pred.shape} vs {target.shape}"
            )
        if sample is None:
            raise ValueError("sample is required for edge-aware loss.")
        pred_dx, pred_dy = self._image_gradients(pred)
        target_dx, target_dy = self._image_gradients(target)
        weights = self._build_loss_weights(
            sample, pred.device, pred.shape[-2], pred.shape[-1]
        )
        weight_dx = 0.5 * (weights[..., :, 1:] + weights[..., :, :-1])
        weight_dy = 0.5 * (weights[..., 1:, :] + weights[..., :-1, :])
        loss_dx = self._weighted_l1(pred_dx, target_dx, weight_dx)
        loss_dy = self._weighted_l1(pred_dy, target_dy, weight_dy)
        return 0.5 * (loss_dx + loss_dy)

    @staticmethod
    def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[..., :, 1:] - image[..., :, :-1]
        dy = image[..., 1:, :] - image[..., :-1, :]
        return dx, dy

    @staticmethod
    def _weighted_l1(
        pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
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
        sky_weight = (
            self._edge_config.sky
            if self._edge_config.sky is not None
            else self._mask_config.sky
        )
        obj_weight = (
            self._edge_config.movable_objects
            if self._edge_config.movable_objects is not None
            else self._mask_config.movable_objects
        )
        if sky_mask is not None:
            weights = torch.where(
                sky_mask.to(device).unsqueeze(0).bool(),
                torch.tensor(sky_weight, device=device),
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
                torch.tensor(obj_weight, device=device),
                weights,
            )
        else:
            raise RuntimeError(
                "object_mask is required but not provided in the sweep sample. "
                "Please run SAM3 preprocessor and generate the masks."
            )
        return weights

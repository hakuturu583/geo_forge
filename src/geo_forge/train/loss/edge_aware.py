from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.loss_base import LossBase


class EdgeAwareLoss(LossBase):
    """
    Compare images using gradient-domain L1 to emphasize edge alignment.
    """

    name = "edge_aware"

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
        pred_dx, pred_dy = self._image_gradients(pred)
        target_dx, target_dy = self._image_gradients(target)
        return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))

    @staticmethod
    def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = image[..., :, 1:] - image[..., :, :-1]
        dy = image[..., 1:, :] - image[..., :-1, :]
        return dx, dy

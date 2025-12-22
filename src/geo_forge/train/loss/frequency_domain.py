from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.loss_base import LossBase


class FrequencyDomainLoss(LossBase):
    """
    Compare images in the frequency domain using log-magnitude L1.
    """

    name = "frequency_domain"

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
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))
        pred_mag = torch.log1p(pred_fft.abs().clamp_min(1e-6))
        target_mag = torch.log1p(target_fft.abs().clamp_min(1e-6))
        return F.l1_loss(pred_mag, target_mag)

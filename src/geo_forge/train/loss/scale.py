from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.config import ScaleLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class ScaleLoss(LossBase):
    """
    Regularize Gaussian scales to avoid over-growing splats.
    """

    name = "scale"

    def __init__(self, config: ScaleLossWeightConfig) -> None:
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
            raise ValueError("sample is required for scale loss.")
        scales = sample.get("gaussian_scales")
        if scales is None:
            raise ValueError("scale loss requires gaussian_scales in sample.")
        if scales.numel() == 0:
            return scales.new_tensor(0.0)
        max_scale = self._config.max_scale
        if max_scale is None:
            return scales.mean()
        if max_scale <= 0:
            raise ValueError("scale loss max_scale must be positive when provided.")
        return F.relu(scales - scales.new_tensor(max_scale)).mean()

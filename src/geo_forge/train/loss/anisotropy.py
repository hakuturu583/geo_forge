from __future__ import annotations

import torch

from geo_forge.train.loss.config import AnisotropyLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class AnisotropyLoss(LossBase):
    """
    Penalize overly anisotropic Gaussian scales.
    """

    name = "anisotropy"

    def __init__(self, config: AnisotropyLossWeightConfig) -> None:
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
            raise ValueError("sample is required for anisotropy loss.")
        scales = sample.get("gaussian_scales")
        if scales is None:
            raise ValueError("anisotropy loss requires gaussian_scales in sample.")
        if scales.numel() == 0:
            return scales.new_tensor(0.0)
        if scales.dim() != 2 or scales.shape[-1] != 3:
            raise ValueError(
                "gaussian_scales must have shape (N, 3); " f"got {tuple(scales.shape)}"
            )
        max_ratio = float(self._config.max_ratio)
        if max_ratio <= 1.0:
            raise ValueError("anisotropy loss max_ratio must be greater than 1.")
        min_scale = scales.min(dim=1).values.clamp_min(1e-6)
        max_scale = scales.max(dim=1).values
        ratio = max_scale / min_scale
        penalty = torch.relu(ratio - max_ratio)
        return penalty.mean()

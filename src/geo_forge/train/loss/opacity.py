from __future__ import annotations

import torch

from geo_forge.train.loss.config import OpacityLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class OpacityLoss(LossBase):
    """
    Regularize Gaussian opacities to discourage high alpha values.
    """

    name = "opacity"

    def __init__(self, config: OpacityLossWeightConfig) -> None:
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
            raise ValueError("sample is required for opacity loss.")
        opacities = sample.get("gaussian_opacities")
        if opacities is None:
            raise ValueError("opacity loss requires gaussian_opacities in sample.")
        if opacities.numel() == 0:
            return opacities.new_tensor(0.0)
        return opacities.abs().mean()

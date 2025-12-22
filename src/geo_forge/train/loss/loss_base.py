from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class LossBase(ABC):
    """
    Base class for loss implementations.
    """

    name: str

    @abstractmethod
    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        """
        Compute the loss for a prediction/target pair.
        """

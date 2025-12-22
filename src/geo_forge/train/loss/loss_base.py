from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from geo_forge.train.loss.config import LossScheduleConfig


class LossBase(ABC):
    """
    Base class for loss implementations.
    """

    name: str
    _schedule: LossScheduleConfig | None
    _last_scale: float | None
    _last_raw: torch.Tensor | None

    def __init__(self, schedule: LossScheduleConfig | None = None) -> None:
        self._schedule = schedule
        self._last_scale = None
        self._last_raw = None

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

    def apply_schedule(
        self, value: torch.Tensor, step: int | None, total_steps: int | None
    ) -> torch.Tensor:
        if self._schedule is None:
            self._last_scale = None
            self._last_raw = None
            return value
        if step is None or total_steps is None:
            scale = 1.0
        else:
            scale = self._schedule.weight_at(step, total_steps)
        self._last_scale = scale
        self._last_raw = value.detach()
        if scale == 1.0:
            return value
        return value * value.new_tensor(scale)

    def log_dict(self, value: torch.Tensor) -> dict[str, float]:
        """
        Build a wandb-friendly log dict for this loss.
        """
        metrics = {f"loss/{self.name}": float(value.item())}
        if self._schedule is not None and self._last_raw is not None:
            metrics[f"loss/{self.name}_raw"] = float(self._last_raw.item())
            scale = 1.0 if self._last_scale is None else self._last_scale
            metrics[f"loss/{self.name}_scale"] = float(scale)
        return metrics

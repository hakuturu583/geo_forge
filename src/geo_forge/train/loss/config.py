from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MaskLossWeightConfig:
    """
    Weights for mask-based photometric losses.
    """

    sky: float = 0.0
    movable_objects: float = 0.1


@dataclass
class FrequencyDomainLossWeightConfig:
    """
    Weights for frequency-domain losses.
    """

    weight: float = 0.0


@dataclass
class EdgeAwareLossWeightConfig:
    """
    Weights for edge-aware losses.
    """

    weight: float = 0.0


@dataclass
class LossScheduleConfig:
    """
    Schedule for scaling the Hausdorff loss over training steps.
    """

    start_scale: float = 1.0
    end_scale: float = 0.5
    start_step: int = 0
    end_step: int | None = None

    def weight_at(self, step: int, total_steps: int) -> float:
        """
        Linearly interpolate the loss scale between start and end.
        """
        if total_steps <= 0:
            return self.end_scale
        if step <= self.start_step:
            return self.start_scale
        end_step = (
            self.end_step if self.end_step is not None else max(total_steps - 1, 0)
        )
        end_step = max(end_step, self.start_step + 1)
        if step >= end_step:
            return self.end_scale
        progress = (step - self.start_step) / float(end_step - self.start_step)
        return self.start_scale + progress * (self.end_scale - self.start_scale)


@dataclass
class HausdorffLossWeightConfig:
    """
    Weights and sampling settings for the Hausdorff loss.
    """

    weight: float = 0.0
    blur: float = 0.05
    max_points: int = 2048
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)


@dataclass
class LossWeightConfig:
    """
    Per-layer loss weights applied to the photometric loss.
    """

    mask: MaskLossWeightConfig = field(default_factory=MaskLossWeightConfig)
    frequency_domain: FrequencyDomainLossWeightConfig = field(
        default_factory=FrequencyDomainLossWeightConfig
    )
    edge_aware: EdgeAwareLossWeightConfig = field(
        default_factory=EdgeAwareLossWeightConfig
    )
    hausdorff: HausdorffLossWeightConfig = field(
        default_factory=HausdorffLossWeightConfig
    )

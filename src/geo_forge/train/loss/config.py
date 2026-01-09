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
class OpacityLossWeightConfig:
    """
    Weight for opacity regularization.
    """

    weight: float = 0.0


@dataclass
class ScaleLossWeightConfig:
    """
    Weight and thresholds for scale regularization.
    """

    weight: float = 0.0
    max_scale: float | None = None


@dataclass
class AnisotropyLossWeightConfig:
    """
    Weight and threshold for anisotropy regularization.
    """

    weight: float = 0.0
    max_ratio: float = 10.0


@dataclass
class SSIMLossWeightConfig:
    """
    Weights and parameters for SSIM losses.
    """

    weight: float = 0.0
    window_size: int = 11
    sigma: float = 1.5
    data_range: float = 1.0


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
class FreeSpaceLossWeightConfig:
    """
    Weights and parameters for the free-space LiDAR loss.
    """

    weight: float = 0.0
    delta: float = 0.3
    n_bins: int = 16
    near: float = 0.1
    far: float = 80.0
    tile_size: int = 16
    packed: bool = False
    debug: bool = False
    downsample_factor: int = 1
    sample_pixels: int | None = None
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)


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
class ChamferLossWeightConfig:
    """
    Weights and sampling settings for the Chamfer loss.
    """

    weight: float = 0.0
    max_points: int = 2048
    max_distance: float = 15.0
    min_points: int = 64
    knn_k: int = 3
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
    opacity: OpacityLossWeightConfig = field(default_factory=OpacityLossWeightConfig)
    scale: ScaleLossWeightConfig = field(default_factory=ScaleLossWeightConfig)
    anisotropy: AnisotropyLossWeightConfig = field(
        default_factory=AnisotropyLossWeightConfig
    )
    free_space: FreeSpaceLossWeightConfig = field(
        default_factory=FreeSpaceLossWeightConfig
    )
    ssim: SSIMLossWeightConfig = field(default_factory=SSIMLossWeightConfig)
    hausdorff: HausdorffLossWeightConfig = field(
        default_factory=HausdorffLossWeightConfig
    )
    chamfer: ChamferLossWeightConfig = field(default_factory=ChamferLossWeightConfig)

from __future__ import annotations

from dataclasses import dataclass, field

from geomloss import SamplesLoss
from geomloss.kernel_samples import kernel_routines
import torch
import torch.nn.functional as F
from torch import nn


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
class HausdorffLossWeightConfig:
    """
    Weights and sampling settings for the Hausdorff loss.
    """

    weight: float = 0.0
    blur: float = 0.05
    max_points: int = 2048
    threshold: float = 0.05


@dataclass
class LossScheduleConfig:
    """
    Schedule for scaling the total loss over training steps.
    """

    start_weight: float = 1.0
    end_weight: float = 0.5
    start_step: int = 0
    end_step: int | None = None

    def weight_at(self, step: int, total_steps: int) -> float:
        """
        Linearly interpolate the loss scale between start and end.
        """
        if total_steps <= 0:
            return self.end_weight
        if step <= self.start_step:
            return self.start_weight
        end_step = (
            self.end_step if self.end_step is not None else max(total_steps - 1, 0)
        )
        end_step = max(end_step, self.start_step + 1)
        if step >= end_step:
            return self.end_weight
        progress = (step - self.start_step) / float(end_step - self.start_step)
        return self.start_weight + progress * (self.end_weight - self.start_weight)


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
    schedule: LossScheduleConfig = field(default_factory=LossScheduleConfig)


def _build_loss_weights(
    sample: dict[str, object],
    loss_weights: LossWeightConfig,
    *,
    device: torch.device,
    height: int,
    width: int,
) -> torch.Tensor:
    sky_mask = sample.get("sky_mask")
    object_mask = sample.get("object_mask")

    weights = torch.ones((1, height, width), device=device)
    if sky_mask is not None:
        weights = torch.where(
            sky_mask.to(device).unsqueeze(0).bool(),
            torch.tensor(loss_weights.mask.sky, device=device),
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
            torch.tensor(loss_weights.mask.movable_objects, device=device),
            weights,
        )
    else:
        raise RuntimeError(
            "object_mask is required but not provided in the sweep sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )
    return weights


def masked_l1_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    sample: dict[str, object],
    loss_weights: LossWeightConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute L1 loss with sky/movable-object masks applied (same weighting as train.py).
    """
    _, _, height, width = pred.unsqueeze(0).shape
    loss_weights = _build_loss_weights(
        sample, loss_weights, device=device, height=height, width=width
    )
    loss_map = F.l1_loss(pred, target, reduction="none")
    weights = loss_weights.expand_as(loss_map).to(loss_map.dtype)
    weight_sum = weights.sum()
    if weight_sum.item() == 0:
        return loss_map.new_tensor(0.0)
    return (loss_map * weights).sum() / weight_sum


def frequency_domain_loss(
    pred: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compare images in the frequency domain using log-magnitude L1.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share the same shape; got {pred.shape} vs {target.shape}"
        )
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))
    pred_mag = torch.log1p(pred_fft.abs().clamp_min(eps))
    target_mag = torch.log1p(target_fft.abs().clamp_min(eps))
    return F.l1_loss(pred_mag, target_mag)


def _image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def edge_aware_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compare images using gradient-domain L1 to emphasize edge alignment.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share the same shape; got {pred.shape} vs {target.shape}"
        )
    pred_dx, pred_dy = _image_gradients(pred)
    target_dx, target_dy = _image_gradients(target)
    return 0.5 * (F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy))


def _sample_image_points(
    image: torch.Tensor, *, max_points: int, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    if image.dim() == 3 and image.shape[0] == 3:
        luminance = (
            0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
        ).clamp_min(0.0)
    elif image.dim() == 2:
        luminance = image.clamp_min(0.0)
    else:
        raise ValueError(
            f"Expected image shape (3, H, W) or (H, W); got {tuple(image.shape)}"
        )

    height, width = luminance.shape
    weights = torch.where(luminance > threshold, luminance, luminance.new_zeros(()))
    flat_weights = weights.flatten()
    if flat_weights.sum().item() <= 0:
        return luminance.new_empty((0, 2)), luminance.new_empty((0,))

    nonzero = torch.nonzero(flat_weights > 0, as_tuple=False).squeeze(1)
    if nonzero.numel() == 0:
        return luminance.new_empty((0, 2)), luminance.new_empty((0,))
    if nonzero.numel() > max_points:
        sampled = torch.multinomial(
            flat_weights[nonzero], num_samples=max_points, replacement=False
        )
        indices = nonzero[sampled]
    else:
        indices = nonzero

    ys = indices // width
    xs = indices % width

    denom_x = max(width - 1, 1)
    denom_y = max(height - 1, 1)
    xs = xs.to(dtype=torch.float32) / float(denom_x)
    ys = ys.to(dtype=torch.float32) / float(denom_y)
    points = torch.stack([xs, ys], dim=1)
    weights = flat_weights[indices].to(dtype=torch.float32)
    weight_sum = weights.sum()
    if weight_sum.item() <= 0:
        return points.new_empty((0, 2)), points.new_empty((0,))
    weights = weights / weight_sum
    return points, weights


def hausdorff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int = 2048,
    threshold: float = 0.05,
    blur: float = 0.05,
) -> torch.Tensor:
    """
    Compute a Hausdorff distance between image-derived point sets.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must share the same shape; got {pred.shape} vs {target.shape}"
        )
    pred_points, pred_weights = _sample_image_points(
        pred, max_points=max_points, threshold=threshold
    )
    target_points, target_weights = _sample_image_points(
        target, max_points=max_points, threshold=threshold
    )
    if pred_points.numel() == 0 or target_points.numel() == 0:
        return pred.new_tensor(0.0)
    loss_fn = SamplesLoss(
        loss="hausdorff", p=2, blur=blur, kernel=kernel_routines["gaussian"]
    )
    return loss_fn(pred_weights, pred_points, target_weights, target_points)


class Loss(nn.Module):
    """
    Composite loss module for masked photometric and auxiliary losses.
    """

    def __init__(self, loss_weights: LossWeightConfig) -> None:
        super().__init__()
        self.loss_weights = loss_weights

    def forward(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object],
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        loss, _ = self.compute(
            pred=pred, target=target, sample=sample, step=step, total_steps=total_steps
        )
        return loss

    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object],
        step: int | None = None,
        total_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute total loss and a dict of weighted component losses.
        """
        device = pred.device
        components: dict[str, torch.Tensor] = {}
        masked = masked_l1_loss(
            pred=pred,
            target=target,
            sample=sample,
            loss_weights=self.loss_weights,
            device=device,
        )
        components["masked_l1"] = masked
        loss = masked

        freq_weight = self.loss_weights.frequency_domain.weight
        if freq_weight > 0:
            freq_loss = frequency_domain_loss(pred, target)
            components["frequency_domain"] = freq_weight * freq_loss
            loss = loss + components["frequency_domain"]

        edge_weight = self.loss_weights.edge_aware.weight
        if edge_weight > 0:
            edge_loss = edge_aware_loss(pred, target)
            components["edge_aware"] = edge_weight * edge_loss
            loss = loss + components["edge_aware"]

        haus_cfg = self.loss_weights.hausdorff
        if haus_cfg.weight > 0:
            haus_loss = hausdorff_loss(
                pred,
                target,
                max_points=haus_cfg.max_points,
                threshold=haus_cfg.threshold,
                blur=haus_cfg.blur,
            )
            components["hausdorff"] = haus_cfg.weight * haus_loss
            loss = loss + components["hausdorff"]

        schedule_weight = self._schedule_weight(step, total_steps)
        if schedule_weight != 1.0:
            scale = loss.new_tensor(schedule_weight)
            loss = loss * scale
            components = {name: value * scale for name, value in components.items()}

        return loss, components

    def _schedule_weight(self, step: int | None, total_steps: int | None) -> float:
        if step is None or total_steps is None:
            return 1.0
        return self.loss_weights.schedule.weight_at(step, total_steps)


def build_wandb_loss_log(
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
) -> dict[str, float]:
    """
    Build a wandb-friendly log dict from loss tensors.
    """
    metrics = {"loss": float(total.item())}
    for name, value in components.items():
        metrics[f"loss/{name}"] = float(value.item())
    return metrics

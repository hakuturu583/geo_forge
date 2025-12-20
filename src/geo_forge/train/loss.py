from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


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
class LossWeightConfig:
    """
    Per-layer loss weights applied to the photometric loss.
    """

    mask: MaskLossWeightConfig = field(default_factory=MaskLossWeightConfig)
    frequency_domain: FrequencyDomainLossWeightConfig = field(
        default_factory=FrequencyDomainLossWeightConfig
    )


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

    loss_weights = torch.ones((1, height, width), device=device)
    if sky_mask is not None:
        loss_weights = torch.where(
            sky_mask.to(device).unsqueeze(0).bool(),
            torch.tensor(loss_weights.mask.sky, device=device),
            loss_weights,
        )
    else:
        raise RuntimeError(
            "sky_mask is required but not provided in the sweep sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )

    if object_mask is not None:
        obj_mask = object_mask.to(device).unsqueeze(0)
        loss_weights = torch.where(
            obj_mask.bool(),
            torch.tensor(loss_weights.mask.movable_objects, device=device),
            loss_weights,
        )
    else:
        raise RuntimeError(
            "object_mask is required but not provided in the sweep sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )
    return loss_weights


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

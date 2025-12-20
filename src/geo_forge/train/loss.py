from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.sharp_based_gs.gs_merge_prune_config import GsMergePruneConfig


def _build_loss_weights(
    sample: dict[str, object],
    config: GsMergePruneConfig,
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
            torch.tensor(config.loss_weights.sky, device=device),
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
            torch.tensor(config.loss_weights.movable_objects, device=device),
            loss_weights,
        )
    else:
        raise RuntimeError(
            "object_mask is required but not provided in the sweep sample. "
            "Please run SAM3 preprocessor and generate the masks."
        )
    return loss_weights


def _masked_l1_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    sample: dict[str, object],
    config: GsMergePruneConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute L1 loss with sky/movable-object masks applied (same weighting as train.py).
    """
    _, _, height, width = pred.unsqueeze(0).shape
    loss_weights = _build_loss_weights(
        sample, config, device=device, height=height, width=width
    )
    loss_map = F.l1_loss(pred, target, reduction="none")
    weights = loss_weights.expand_as(loss_map).to(loss_map.dtype)
    weight_sum = weights.sum()
    if weight_sum.item() == 0:
        return loss_map.new_tensor(0.0)
    return (loss_map * weights).sum() / weight_sum

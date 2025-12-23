from __future__ import annotations

import torch
from torch import nn

from geo_forge.train.loss.config import LossWeightConfig
from geo_forge.train.loss.chamfer import ChamferLoss
from geo_forge.train.loss.edge_aware import EdgeAwareLoss
from geo_forge.train.loss.frequency_domain import FrequencyDomainLoss
from geo_forge.train.loss.hausdorff import HausdorffLoss
from geo_forge.train.loss.masked_l1 import MaskedL1Loss
from geo_forge.train.loss.ssim import SSIMLoss


class Loss(nn.Module):
    """
    Composite loss module for masked photometric and auxiliary losses.
    """

    def __init__(self, loss_weights: LossWeightConfig) -> None:
        super().__init__()
        self.loss_weights = loss_weights
        self._masked_l1 = MaskedL1Loss(loss_weights.mask)
        self._frequency_domain = FrequencyDomainLoss()
        self._edge_aware = EdgeAwareLoss()
        self._ssim = SSIMLoss(loss_weights.ssim)
        self._hausdorff = HausdorffLoss(loss_weights.hausdorff)
        self._chamfer = ChamferLoss(loss_weights.chamfer)
        self._losses = {
            self._masked_l1.name: self._masked_l1,
            self._frequency_domain.name: self._frequency_domain,
            self._edge_aware.name: self._edge_aware,
            self._ssim.name: self._ssim,
            self._hausdorff.name: self._hausdorff,
            self._chamfer.name: self._chamfer,
        }

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
        components: dict[str, torch.Tensor] = {}
        masked = self._masked_l1.compute(
            pred=pred,
            target=target,
            sample=sample,
            step=step,
            total_steps=total_steps,
        )
        components["masked_l1"] = masked
        loss = masked

        freq_weight = self.loss_weights.frequency_domain.weight
        if freq_weight > 0:
            freq_loss = self._frequency_domain.compute(
                pred=pred,
                target=target,
                sample=sample,
                step=step,
                total_steps=total_steps,
            )
            components["frequency_domain"] = freq_weight * freq_loss
            loss = loss + components["frequency_domain"]

        edge_weight = self.loss_weights.edge_aware.weight
        if edge_weight > 0:
            edge_loss = self._edge_aware.compute(
                pred=pred,
                target=target,
                sample=sample,
                step=step,
                total_steps=total_steps,
            )
            components["edge_aware"] = edge_weight * edge_loss
            loss = loss + components["edge_aware"]

        ssim_cfg = self.loss_weights.ssim
        if ssim_cfg.weight > 0:
            ssim_loss = self._ssim.compute(
                pred=pred,
                target=target,
                sample=sample,
                step=step,
                total_steps=total_steps,
            )
            components["ssim"] = ssim_cfg.weight * ssim_loss
            loss = loss + components["ssim"]

        haus_cfg = self.loss_weights.hausdorff
        if haus_cfg.weight > 0:
            haus_loss = self._hausdorff.compute(
                pred=pred,
                target=target,
                sample=sample,
                step=step,
                total_steps=total_steps,
            )
            components["hausdorff"] = haus_cfg.weight * haus_loss
            loss = loss + components["hausdorff"]

        chamfer_cfg = self.loss_weights.chamfer
        if chamfer_cfg.weight > 0:
            chamfer_loss = self._chamfer.compute(
                pred=pred,
                target=target,
                sample=sample,
                step=step,
                total_steps=total_steps,
            )
            components["chamfer"] = chamfer_cfg.weight * chamfer_loss
            loss = loss + components["chamfer"]

        return loss, components

    def build_wandb_log(
        self, total: torch.Tensor, components: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        """
        Build a wandb-friendly log dict from loss tensors.
        """
        metrics = {"loss/total": float(total.item())}
        for name, value in components.items():
            loss_fn = self._losses.get(name)
            if loss_fn is None:
                metrics[f"loss/{name}"] = float(value.item())
            else:
                metrics.update(loss_fn.log_dict(value))
        return metrics

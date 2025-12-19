from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove

Params = Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]


@dataclass
class MergePruneStrategy(DefaultStrategy):
    """
    DefaultStrategy with an extra merge stage that periodically absorbs nearby Gaussians
    into a representative and prunes the rest.

    Candidates are bucketed by voxel; within each voxel, we pick a representative and
    absorb close neighbors, then remove the merged entries using gsplat's `remove`.
    """

    merge_start_iter: int = 0
    merge_stop_iter: int = 10000000
    merge_every: int = 50

    voxel_size: float = 0.1
    merge_radius: float = 0.05

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Params,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        super().step_post_backward(params, optimizers, state, step, info, packed=packed)

        if not (self.merge_start_iter <= step < self.merge_stop_iter):
            return
        if self.merge_every <= 0 or step % self.merge_every != 0:
            return

        n_merged = self._merge_close_gaussians(params, optimizers, state)
        if self.verbose and n_merged:
            print(
                f"[MergePruneStrategy] step={step}: merged/pruned {n_merged} gaussians. "
                f"now N={len(params['means'])}"
            )

    @torch.no_grad()
    def _merge_close_gaussians(
        self,
        params: Params,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        means = params["means"]  # [N, 3]
        device = means.device
        n_points = int(means.shape[0])
        if n_points < 2:
            return 0

        opacities = torch.sigmoid(params["opacities"].flatten())  # [N]

        scene_scale = float(state.get("scene_scale", 1.0))
        normalized = means / max(scene_scale, 1e-8)

        voxel = torch.floor(normalized / float(self.voxel_size)).to(torch.int32)
        key = (
            (voxel[:, 0] * 73856093)
            ^ (voxel[:, 1] * 19349663)
            ^ (voxel[:, 2] * 83492791)
        )  # large primes to hash 3D voxel coords

        order = torch.argsort(key)
        key_sorted = key[order]
        boundaries = torch.nonzero(key_sorted[1:] != key_sorted[:-1]).flatten() + 1
        starts = torch.cat(
            [torch.zeros(1, device=device, dtype=torch.long), boundaries]
        )
        ends = torch.cat(
            [boundaries, torch.tensor([n_points], device=device, dtype=torch.long)]
        )

        remove_mask = torch.zeros(n_points, device=device, dtype=torch.bool)

        for start, end in zip(starts.tolist(), ends.tolist()):
            idx = order[start:end]
            if idx.numel() <= 1:
                continue

            rep_local = torch.argmax(opacities[idx])
            rep = idx[rep_local]

            dist = torch.norm(normalized[idx] - normalized[rep], dim=-1)
            merge_sel = dist < float(self.merge_radius)
            merge_sel[rep_local] = False
            to_merge = idx[merge_sel]
            if to_merge.numel() == 0:
                continue

            w = opacities[to_merge].clamp_min(1e-6)
            w_rep = opacities[rep].clamp_min(1e-6)
            denom = w_rep + w.sum()

            new_mean = (
                w_rep * means[rep] + (w[:, None] * means[to_merge]).sum(dim=0)
            ) / denom
            means[rep].copy_(new_mean)

            if "scales" in params:
                new_scales = (
                    w_rep * params["scales"][rep]
                    + (w[:, None] * params["scales"][to_merge]).sum(dim=0)
                ) / denom
                params["scales"][rep].copy_(new_scales)

            if "colors" in params:
                new_colors = (
                    w_rep * params["colors"][rep]
                    + (w[:, None] * params["colors"][to_merge]).sum(dim=0)
                ) / denom
                params["colors"][rep].copy_(new_colors)

            max_opa = torch.maximum(opacities[rep], opacities[to_merge].max())
            params["opacities"][rep].copy_(torch.logit(max_opa.clamp(1e-6, 1.0 - 1e-6)))

            remove_mask[to_merge] = True

        n_remove = int(remove_mask.sum().item())
        if n_remove == 0:
            return 0

        remove(params=params, optimizers=optimizers, state=state, mask=remove_mask)
        return n_remove

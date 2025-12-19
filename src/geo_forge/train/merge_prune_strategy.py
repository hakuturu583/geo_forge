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
        group_id_sorted = torch.zeros(n_points, device=device, dtype=torch.long)
        group_id_sorted[1:] = torch.cumsum(
            (key_sorted[1:] != key_sorted[:-1]).to(torch.long), dim=0
        )
        n_groups = int(group_id_sorted[-1].item() + 1)
        if n_groups == n_points:
            return 0

        op_sorted = opacities[order]
        max_opa = torch.zeros(n_groups, device=device, dtype=opacities.dtype)
        max_opa.scatter_reduce_(0, group_id_sorted, op_sorted, reduce="amax")
        is_max = op_sorted == max_opa[group_id_sorted]

        pos = torch.arange(n_points, device=device)
        pos_candidates = torch.where(is_max, pos, torch.full_like(pos, n_points))
        # Pick the first max in key-sorted order to match torch.argmax tie-breaks.
        rep_pos = torch.full((n_groups,), n_points, device=device, dtype=torch.long)
        rep_pos.scatter_reduce_(0, group_id_sorted, pos_candidates, reduce="amin")
        rep_idx = order[rep_pos]
        rep_pos_per_sorted = rep_pos[group_id_sorted]
        rep_idx_per_sorted = order[rep_pos_per_sorted]

        rep_of_point = torch.empty(n_points, device=device, dtype=torch.long)
        rep_of_point[order] = rep_idx_per_sorted
        group_id = torch.empty(n_points, device=device, dtype=torch.long)
        group_id[order] = group_id_sorted

        idx = torch.arange(n_points, device=device)
        dist = torch.norm(normalized - normalized[rep_of_point], dim=-1)
        merge_sel = (dist < float(self.merge_radius)) & (idx != rep_of_point)

        merge_counts = torch.zeros(n_groups, device=device, dtype=torch.int32)
        merge_counts.scatter_add_(0, group_id, merge_sel.to(torch.int32))
        has_merge = merge_counts > 0
        if not bool(has_merge.any()):
            return 0

        merge_or_rep = merge_sel | (idx == rep_of_point)
        weights = opacities.clamp_min(1e-6) * merge_or_rep

        sum_w = torch.zeros(n_groups, device=device, dtype=means.dtype)
        sum_w.scatter_add_(0, group_id, weights)

        sum_wx = torch.zeros(n_groups, 3, device=device, dtype=means.dtype)
        sum_wx.scatter_add_(
            0, group_id[:, None].expand(-1, 3), weights[:, None] * means
        )
        new_means = sum_wx / sum_w[:, None]

        rep_to_update = rep_idx[has_merge]
        means.index_copy_(0, rep_to_update, new_means[has_merge])

        if "scales" in params:
            scales = params["scales"]
            sum_ws = torch.zeros(
                n_groups, scales.shape[1], device=device, dtype=scales.dtype
            )
            sum_ws.scatter_add_(
                0,
                group_id[:, None].expand(-1, scales.shape[1]),
                weights[:, None] * scales,
            )
            new_scales = sum_ws / sum_w[:, None]
            scales.index_copy_(0, rep_to_update, new_scales[has_merge])

        if "colors" in params:
            colors = params["colors"]
            sum_wc = torch.zeros(
                n_groups, colors.shape[1], device=device, dtype=colors.dtype
            )
            sum_wc.scatter_add_(
                0,
                group_id[:, None].expand(-1, colors.shape[1]),
                weights[:, None] * colors,
            )
            new_colors = sum_wc / sum_w[:, None]
            colors.index_copy_(0, rep_to_update, new_colors[has_merge])

        masked_opa = torch.where(merge_or_rep, opacities, torch.zeros_like(opacities))
        max_opa_merge = torch.zeros(n_groups, device=device, dtype=opacities.dtype)
        max_opa_merge.scatter_reduce_(0, group_id, masked_opa, reduce="amax")
        new_opacities = torch.logit(max_opa_merge.clamp(1e-6, 1.0 - 1e-6))
        params["opacities"].view(-1).index_copy_(
            0, rep_to_update, new_opacities[has_merge]
        )

        remove_mask = merge_sel

        n_remove = int(remove_mask.sum().item())
        if n_remove == 0:
            return 0

        remove(params=params, optimizers=optimizers, state=state, mask=remove_mask)
        return n_remove

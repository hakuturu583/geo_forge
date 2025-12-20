from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Union

import torch

from gsplat.strategy.default import DefaultStrategy
from gsplat.strategy.ops import remove

Params = Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]


@dataclass
class BackfacePruneConfig:
    """
    Configuration for pruning low-impact gaussians that stay behind the camera.
    """

    enabled: bool = False
    min_steps: int = 100  # Number of consecutive steps a gaussian must qualify.
    opacity_threshold: float = 0.01  # Opacity cutoff for "low impact" classification.
    radii_threshold: float = 1.0  # Projected radius cutoff for "low impact".
    depth_threshold: float = 0.0  # Depth cutoff for backface/behind-camera pruning.
    border: float = 2.0  # Pixel border margin when classifying offscreen.


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
    merge_every: int = 1000

    voxel_size: float = 0.1
    merge_radius: float = 0.05
    prune_scale_threshold: float = 0.5
    backface_prune: BackfacePruneConfig = field(default_factory=BackfacePruneConfig)

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

        n_merged = self._merge_close_gaussians(params, optimizers, state, info)
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
        info: Dict[str, Any],
    ) -> int:
        means = params["means"]  # [N, 3]
        device = means.device
        n_points = int(means.shape[0])
        if n_points < 2:
            return 0

        opacities = torch.sigmoid(params["opacities"].flatten())  # [N]
        scene_scale = float(state.get("scene_scale", 1.0))
        normalized = means / max(scene_scale, 1e-8)
        prune_mask = self._compute_prune_mask(
            params, opacities, normalized, info=info, state=state
        )
        has_prune = bool(prune_mask.any())

        order, group_id_sorted, group_id, n_groups = self._group_by_voxel(
            normalized, float(self.voxel_size)
        )
        if n_groups == n_points:
            if not has_prune:
                return 0
            n_remove = int(prune_mask.sum().item())
            if n_remove == 0:
                return 0
            remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
            self._update_backface_counts_after_remove(state, prune_mask)
            return n_remove

        rep_idx, rep_of_point = self._select_representatives(
            order, group_id_sorted, opacities, n_groups
        )

        idx = torch.arange(n_points, device=device)
        dist = torch.norm(normalized - normalized[rep_of_point], dim=-1)
        merge_sel = (dist < float(self.merge_radius)) & (idx != rep_of_point)

        merge_counts = torch.zeros(n_groups, device=device, dtype=torch.int32)
        merge_counts.scatter_add_(0, group_id, merge_sel.to(torch.int32))
        has_merge = merge_counts > 0
        if not bool(has_merge.any()):
            if not has_prune:
                return 0
            n_remove = int(prune_mask.sum().item())
            if n_remove == 0:
                return 0
            remove(params=params, optimizers=optimizers, state=state, mask=prune_mask)
            self._update_backface_counts_after_remove(state, prune_mask)
            return n_remove

        self._apply_merge_updates(
            params,
            opacities,
            means,
            group_id,
            rep_idx,
            rep_of_point,
            merge_sel,
            has_merge,
            n_groups,
        )

        remove_mask = merge_sel | prune_mask

        n_remove = int(remove_mask.sum().item())
        if n_remove == 0:
            return 0

        remove(params=params, optimizers=optimizers, state=state, mask=remove_mask)
        self._update_backface_counts_after_remove(state, remove_mask)
        return n_remove

    def _compute_prune_mask(
        self,
        params: Params,
        opacities: torch.Tensor,
        normalized: torch.Tensor,
        info: Dict[str, Any] | None,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        n_points = int(opacities.shape[0])
        prune_mask = torch.zeros(n_points, device=opacities.device, dtype=torch.bool)
        prune_opa = getattr(self, "prune_opa", None)
        if prune_opa is not None:
            prune_mask |= opacities < float(prune_opa)
        if "scales" in params and self.prune_scale_threshold > 0:
            scales = torch.exp(params["scales"])
            max_scales = scales.max(dim=1).values
            prune_mask |= max_scales > float(self.prune_scale_threshold)
        prune_mask |= self._singleton_voxel_mask(normalized, voxel_size=1.0)
        prune_mask |= self._compute_backface_prune_mask(opacities, info, state)
        return prune_mask

    def _compute_backface_prune_mask(
        self,
        opacities: torch.Tensor,
        info: Dict[str, Any] | None,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        n_points = int(opacities.shape[0])
        device = opacities.device
        if not self.backface_prune.enabled or info is None or n_points == 0:
            return torch.zeros(n_points, device=device, dtype=torch.bool)

        cfg = self.backface_prune
        means2d = info.get("means2d")
        radii = info.get("radii")
        depths = info.get("depths")
        width = info.get("width")
        height = info.get("height")
        if (
            means2d is None
            or radii is None
            or depths is None
            or width is None
            or height is None
        ):
            return torch.zeros(n_points, device=device, dtype=torch.bool)

        if means2d.dim() == 2:
            means2d = means2d.unsqueeze(0)
        if radii.dim() == 3:
            radii = radii.max(dim=-1).values
        elif radii.dim() == 2 and radii.shape[-1] == 2:
            radii = radii.max(dim=-1).values
        if radii.dim() == 1:
            radii = radii.unsqueeze(0)
        if depths.dim() == 2 and depths.shape[-1] == 1:
            depths = depths.squeeze(-1)
        if depths.dim() == 1:
            depths = depths.unsqueeze(0)

        if means2d.shape[1] != n_points:
            return torch.zeros(n_points, device=device, dtype=torch.bool)

        border = float(cfg.border)
        x = means2d[..., 0]
        y = means2d[..., 1]
        offscreen = (
            (x < -border)
            | (x > float(width) + border)
            | (y < -border)
            | (y > float(height) + border)
        )
        backface = depths <= float(cfg.depth_threshold)
        low_radii = radii <= float(cfg.radii_threshold)
        low_opa = opacities <= float(cfg.opacity_threshold)
        low_impact = low_radii | low_opa.unsqueeze(0)
        candidate = (offscreen | backface) & low_impact
        candidate_all = candidate.all(dim=0)

        counts = state.get("backface_prune_counts")
        if (
            counts is None
            or not torch.is_tensor(counts)
            or counts.shape[0] != n_points
            or counts.device != device
        ):
            counts = torch.zeros(n_points, device=device, dtype=torch.int32)
        counts = torch.where(candidate_all, counts + 1, torch.zeros_like(counts))
        state["backface_prune_counts"] = counts

        min_steps = max(1, int(cfg.min_steps))
        return counts >= min_steps

    def _update_backface_counts_after_remove(
        self, state: Dict[str, Any], remove_mask: torch.Tensor
    ) -> None:
        counts = state.get("backface_prune_counts")
        if counts is None or not torch.is_tensor(counts):
            return
        if counts.shape[0] != remove_mask.shape[0]:
            return
        state["backface_prune_counts"] = counts[~remove_mask]

    def _singleton_voxel_mask(
        self, normalized: torch.Tensor, voxel_size: float
    ) -> torch.Tensor:
        order, group_id_sorted, _, n_groups = self._group_by_voxel(
            normalized, voxel_size
        )
        counts = torch.zeros(n_groups, device=normalized.device, dtype=torch.int32)
        counts.scatter_add_(
            0, group_id_sorted, torch.ones_like(group_id_sorted, dtype=torch.int32)
        )
        single_sorted = counts[group_id_sorted] < 2
        single = torch.empty(
            normalized.shape[0], device=normalized.device, dtype=torch.bool
        )
        single[order] = single_sorted
        return single

    def _group_by_voxel(
        self, normalized: torch.Tensor, voxel_size: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        voxel = torch.floor(normalized / float(voxel_size)).to(torch.int32)
        key = self._voxel_hash(voxel)
        order = torch.argsort(key)
        key_sorted = key[order]
        group_id_sorted = torch.zeros(
            normalized.shape[0], device=normalized.device, dtype=torch.long
        )
        group_id_sorted[1:] = torch.cumsum(
            (key_sorted[1:] != key_sorted[:-1]).to(torch.long), dim=0
        )
        n_groups = int(group_id_sorted[-1].item() + 1)
        group_id = torch.empty_like(group_id_sorted)
        group_id[order] = group_id_sorted
        return order, group_id_sorted, group_id, n_groups

    def _voxel_hash(self, voxel: torch.Tensor) -> torch.Tensor:
        return (
            (voxel[:, 0] * 73856093)
            ^ (voxel[:, 1] * 19349663)
            ^ (voxel[:, 2] * 83492791)
        )

    def _select_representatives(
        self,
        order: torch.Tensor,
        group_id_sorted: torch.Tensor,
        opacities: torch.Tensor,
        n_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        op_sorted = opacities[order]
        max_opa = torch.zeros(n_groups, device=opacities.device, dtype=opacities.dtype)
        max_opa.scatter_reduce_(0, group_id_sorted, op_sorted, reduce="amax")
        is_max = op_sorted == max_opa[group_id_sorted]

        n_points = int(opacities.shape[0])
        pos = torch.arange(n_points, device=opacities.device)
        pos_candidates = torch.where(is_max, pos, torch.full_like(pos, n_points))
        # Pick the first max in key-sorted order to match torch.argmax tie-breaks.
        rep_pos = torch.full(
            (n_groups,), n_points, device=opacities.device, dtype=torch.long
        )
        rep_pos.scatter_reduce_(0, group_id_sorted, pos_candidates, reduce="amin")
        rep_idx = order[rep_pos]
        rep_pos_per_sorted = rep_pos[group_id_sorted]
        rep_idx_per_sorted = order[rep_pos_per_sorted]

        rep_of_point = torch.empty(n_points, device=opacities.device, dtype=torch.long)
        rep_of_point[order] = rep_idx_per_sorted
        return rep_idx, rep_of_point

    def _apply_merge_updates(
        self,
        params: Params,
        opacities: torch.Tensor,
        means: torch.Tensor,
        group_id: torch.Tensor,
        rep_idx: torch.Tensor,
        rep_of_point: torch.Tensor,
        merge_sel: torch.Tensor,
        has_merge: torch.Tensor,
        n_groups: int,
    ) -> None:
        idx = torch.arange(opacities.shape[0], device=opacities.device)
        merge_or_rep = merge_sel | (idx == rep_of_point)
        weights = opacities.clamp_min(1e-6) * merge_or_rep

        sum_w = torch.zeros(n_groups, device=means.device, dtype=means.dtype)
        sum_w.scatter_add_(0, group_id, weights)

        sum_wx = torch.zeros(n_groups, 3, device=means.device, dtype=means.dtype)
        sum_wx.scatter_add_(
            0, group_id[:, None].expand(-1, 3), weights[:, None] * means
        )
        new_means = sum_wx / sum_w[:, None]

        rep_to_update = rep_idx[has_merge]
        means.index_copy_(0, rep_to_update, new_means[has_merge])

        if "scales" in params:
            scales = params["scales"]
            sum_ws = torch.zeros(
                n_groups, scales.shape[1], device=means.device, dtype=scales.dtype
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
                n_groups, colors.shape[1], device=means.device, dtype=colors.dtype
            )
            sum_wc.scatter_add_(
                0,
                group_id[:, None].expand(-1, colors.shape[1]),
                weights[:, None] * colors,
            )
            new_colors = sum_wc / sum_w[:, None]
            colors.index_copy_(0, rep_to_update, new_colors[has_merge])

        masked_opa = torch.where(merge_or_rep, opacities, torch.zeros_like(opacities))
        max_opa_merge = torch.zeros(
            n_groups, device=opacities.device, dtype=opacities.dtype
        )
        max_opa_merge.scatter_reduce_(0, group_id, masked_opa, reduce="amax")
        new_opacities = torch.logit(max_opa_merge.clamp(1e-6, 1.0 - 1e-6))
        params["opacities"].view(-1).index_copy_(
            0, rep_to_update, new_opacities[has_merge]
        )

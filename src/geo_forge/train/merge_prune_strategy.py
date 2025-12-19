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
    merge_radius: float = 0.3
    color_threshold: float = 0.2  # max L2 distance in color space to merge
    quat_angle_threshold: float = 0.35  # radians; ~20 deg max orientation delta

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
        colors_all = params["colors"]
        quats_all = params["quats"]

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

            if merge_sel.any():
                # Filter by color proximity.
                rep_color = colors_all[rep]
                color_dist = torch.norm(colors_all[idx] - rep_color, dim=-1)
                merge_sel &= color_dist <= float(self.color_threshold)

            if merge_sel.any():
                # Filter by orientation proximity (pre-normalize rep outside loop).
                rep_quat = quats_all[rep]
                rep_quat = rep_quat / torch.clamp(rep_quat.norm(), min=1e-12)

                cand_quats = quats_all[idx]
                cand_quats = cand_quats / torch.clamp(
                    cand_quats.norm(dim=-1, keepdim=True), min=1e-12
                )
                dot = torch.sum(rep_quat[None, :] * cand_quats, dim=-1)
                aligned_dot = dot.abs()
                angles = 2.0 * torch.acos(torch.clamp(aligned_dot, max=1.0))
                merge_sel &= angles <= float(self.quat_angle_threshold)
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

            # Slerp representative quaternion toward weighted mean of candidates.
            weight_frac = float(w.sum() / denom)
            if weight_frac > 0.0:
                # Reuse already-normalized rep_quat and cand_quats to reduce overhead.
                aligned_quats = torch.where(
                    (dot[merge_sel, None] < 0.0),
                    -cand_quats[merge_sel],
                    cand_quats[merge_sel],
                )
                target_quat = aligned_quats.mul(w[:, None]).sum(dim=0)
                target_quat = target_quat / torch.clamp(
                    target_quat.norm(), min=1e-12
                )
                new_quat = self._slerp(rep_quat, target_quat, weight_frac)
                params["quats"][rep].copy_(new_quat)

            remove_mask[to_merge] = True

        n_remove = int(remove_mask.sum().item())
        if n_remove == 0:
            return 0

        remove(params=params, optimizers=optimizers, state=state, mask=remove_mask)
        return n_remove

    @staticmethod
    def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        """
        Spherical linear interpolation between two unit quaternions.
        """
        q0_n = q0 / torch.clamp(q0.norm(), min=1e-12)
        q1_n = q1 / torch.clamp(q1.norm(), min=1e-12)
        dot = torch.sum(q0_n * q1_n)
        if dot < 0.0:
            q1_n = -q1_n
            dot = -dot
        dot = torch.clamp(dot, max=1.0)
        if dot > 0.9995:
            result = q0_n + t * (q1_n - q0_n)
            return result / torch.clamp(result.norm(), min=1e-12)
        theta_0 = torch.acos(dot)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)
        sin_theta_0 = torch.sin(theta_0)
        s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0_n + s1 * q1_n

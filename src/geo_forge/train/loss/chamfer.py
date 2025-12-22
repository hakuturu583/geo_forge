from __future__ import annotations

import torch

from geo_forge.train.loss.config import ChamferLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class ChamferLoss(LossBase):
    """
    Compute a symmetric Chamfer distance between Gaussian mean point sets.
    """

    name = "chamfer"

    def __init__(self, config: ChamferLossWeightConfig) -> None:
        super().__init__(config.schedule)
        self._config = config

    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        if sample is None:
            raise ValueError("sample is required for Chamfer loss.")
        means = sample.get("gaussian_means")
        init_means = sample.get("init_gaussian_means")
        intrinsics = sample.get("intrinsics")
        c2w = sample.get("c2w")
        width = sample.get("width")
        height = sample.get("height")
        if (
            means is None
            or init_means is None
            or intrinsics is None
            or c2w is None
            or width is None
            or height is None
        ):
            raise ValueError(
                "Chamfer loss requires gaussian_means, init_gaussian_means, "
                "intrinsics, c2w, width, and height in sample."
            )
        if means.dim() != 2 or means.shape[-1] != 3:
            raise ValueError(
                "gaussian_means must have shape (N, 3); " f"got {tuple(means.shape)}"
            )
        if init_means.dim() != 2 or init_means.shape[-1] != 3:
            raise ValueError(
                "init_gaussian_means must have shape (N, 3); "
                f"got {tuple(init_means.shape)}"
            )

        means = means.to(device=means.device, dtype=torch.float32)
        init_means = init_means.to(device=means.device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=means.device, dtype=torch.float32)
        c2w = c2w.to(device=means.device, dtype=torch.float32)
        pred_points = self._filter_in_view(
            means,
            intrinsics=intrinsics,
            c2w=c2w,
            width=int(width),
            height=int(height),
        )
        target_points = self._filter_in_view(
            init_means,
            intrinsics=intrinsics,
            c2w=c2w,
            width=int(width),
            height=int(height),
        )
        pred_points = self._sample_points(
            pred_points, max_points=self._config.max_points
        )
        target_points = self._sample_points(
            target_points, max_points=self._config.max_points
        )
        if pred_points.numel() == 0 or target_points.numel() == 0:
            return self.apply_schedule(pred.new_tensor(0.0), step, total_steps)

        dist = torch.cdist(pred_points, target_points, p=2)
        loss = 0.5 * (dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())
        diag = self._aabb_diag(init_means)
        if diag > 0:
            loss = loss / loss.new_tensor(diag)
        else:
            loss = loss.new_tensor(0.0)
        return self.apply_schedule(loss, step, total_steps)

    @staticmethod
    def _filter_in_view(
        points: torch.Tensor,
        *,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        if points.numel() == 0:
            return points
        w2c = torch.inverse(c2w)
        rot = w2c[:3, :3]
        trans = w2c[:3, 3]
        points_cam = points @ rot.T + trans
        z = points_cam[:, 2]
        in_front = z > 0
        if not torch.any(in_front):
            return points.new_empty((0, 3))
        points_cam = points_cam[in_front]
        z = z[in_front]
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        u = fx * (points_cam[:, 0] / z) + cx
        v = fy * (points_cam[:, 1] / z) + cy
        in_bounds = (u >= 0.0) & (u < float(width)) & (v >= 0.0) & (v < float(height))
        if not torch.any(in_bounds):
            return points.new_empty((0, 3))
        return points[in_front][in_bounds]

    @staticmethod
    def _sample_points(points: torch.Tensor, *, max_points: int) -> torch.Tensor:
        if max_points <= 0:
            raise ValueError("max_points must be positive.")
        num_points = points.shape[0]
        if num_points <= max_points:
            return points
        indices = torch.randperm(num_points, device=points.device)[:max_points]
        return points[indices]

    @staticmethod
    def _aabb_diag(points: torch.Tensor) -> float:
        if points.numel() == 0:
            return 0.0
        min_vals = points.min(dim=0).values
        max_vals = points.max(dim=0).values
        diag = (max_vals - min_vals).norm().item()
        return float(diag)

from __future__ import annotations

import math

import gsplat
import torch

from geo_forge.train.loss.config import FreeSpaceLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class FreeSpaceLoss(LossBase):
    """
    Penalize alpha accumulation before LiDAR surface depths.
    """

    name = "free_space"

    def __init__(self, config: FreeSpaceLossWeightConfig) -> None:
        super().__init__()
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
            raise ValueError("sample is required for free-space loss.")
        means = sample.get("gaussian_means")
        quats = sample.get("gaussian_quats")
        scales = sample.get("gaussian_scales")
        opacities = sample.get("gaussian_opacities")
        intrinsics = sample.get("intrinsics")
        c2w = sample.get("c2w")
        lidar_depth = sample.get("lidar_depth")
        valid_mask = sample.get("lidar_valid")
        sky_mask = sample.get("sky_mask")
        object_mask = sample.get("object_mask")
        if (
            means is None
            or quats is None
            or scales is None
            or opacities is None
            or intrinsics is None
            or c2w is None
        ):
            raise ValueError(
                "Free-space loss requires gaussian_means, gaussian_quats, "
                "gaussian_scales, gaussian_opacities, intrinsics, c2w, "
                "and lidar_depth in sample."
            )
        if lidar_depth is None:
            self._log(
                "missing_lidar_depth",
                "lidar_depth is missing; returning 0.",
                sample=sample,
                step=step,
            )
            return means.new_tensor(0.0)

        if means.numel() == 0:
            self._log("empty_means", "gaussian_means is empty; returning 0.", step=step)
            return means.new_tensor(0.0)
        if means.dim() != 2 or means.shape[-1] != 3:
            raise ValueError(
                "gaussian_means must have shape (N, 3); " f"got {tuple(means.shape)}"
            )
        if quats.dim() != 2 or quats.shape[-1] != 4:
            raise ValueError(
                "gaussian_quats must have shape (N, 4); " f"got {tuple(quats.shape)}"
            )
        if scales.dim() != 2 or scales.shape[-1] != 3:
            raise ValueError(
                "gaussian_scales must have shape (N, 3); " f"got {tuple(scales.shape)}"
            )
        if opacities.dim() == 2 and opacities.shape[-1] == 1:
            opacities = opacities.squeeze(-1)
        if opacities.dim() != 1:
            raise ValueError(
                "gaussian_opacities must have shape (N,) or (N, 1); "
                f"got {tuple(opacities.shape)}"
            )

        device = means.device
        dtype = means.dtype
        depth = lidar_depth.to(device=device, dtype=dtype)
        if depth.dim() == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.dim() != 2:
            raise ValueError(
                f"lidar_depth must have shape (H, W); got {tuple(depth.shape)}"
            )
        height, width = int(depth.shape[0]), int(depth.shape[1])
        if "width" in sample and int(sample["width"]) != width:
            raise ValueError(
                f"lidar_depth width {width} does not match sample width {sample['width']}"
            )
        if "height" in sample and int(sample["height"]) != height:
            raise ValueError(
                f"lidar_depth height {height} does not match sample height {sample['height']}"
            )

        valid: torch.Tensor
        if valid_mask is None:
            valid = torch.ones((height, width), device=device, dtype=torch.bool)
        else:
            valid = (
                valid_mask.to(device=device)
                if isinstance(valid_mask, torch.Tensor)
                else torch.as_tensor(valid_mask, device=device)
            )
            if valid.dim() == 3 and valid.shape[0] == 1:
                valid = valid[0]
            if valid.shape != depth.shape:
                raise ValueError(
                    "lidar_valid must match lidar_depth shape "
                    f"{tuple(depth.shape)}; got {tuple(valid.shape)}"
                )
            valid = valid.bool()

        valid = valid & self._exclude_mask(sky_mask, depth.shape, device)
        valid = valid & self._exclude_mask(object_mask, depth.shape, device)

        near = float(self._config.near)
        far = float(self._config.far)
        finite = torch.isfinite(depth)
        in_range = (depth > near) & (depth < far)
        valid = valid & finite & in_range
        if valid.sum() == 0:
            self._log(
                "no_valid_depth",
                "no valid depth pixels after masking; returning 0.",
                sample=sample,
                step=step,
                extra={
                    "finite": int(finite.sum().item()),
                    "in_range": int(in_range.sum().item()),
                    "valid_mask": int(valid_mask.sum().item())
                    if isinstance(valid_mask, torch.Tensor)
                    else None,
                },
            )
            return means.new_tensor(0.0)

        delta = float(self._config.delta)
        depth_thr = (depth - delta).clamp(min=near, max=far)
        bins = torch.linspace(
            near, far, steps=self._config.n_bins, device=device, dtype=dtype
        )
        bin_idx = torch.bucketize(depth_thr.reshape(-1), bins) - 1
        bin_idx = bin_idx.clamp(min=0, max=self._config.n_bins - 1).reshape(
            height, width
        )

        intrinsics_t = intrinsics.to(device=device, dtype=dtype)
        c2w_t = c2w.to(device=device, dtype=dtype)
        viewmat = torch.inverse(c2w_t)
        z_cam = self._camera_z(viewmat, means)

        (radii, means2d, depths, conics, _) = gsplat.rendering.fully_fused_projection(
            means=means,
            covars=None,
            quats=quats,
            scales=scales,
            viewmats=viewmat[None, ...],
            Ks=intrinsics_t[None, ...],
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            opacities=opacities,
            packed=self._config.packed,
        )

        tile_size = int(self._config.tile_size)
        tile_width = math.ceil(width / tile_size)
        tile_height = math.ceil(height / tile_size)
        _, isect_ids, flatten_ids = gsplat.rendering.isect_tiles(
            means2d=means2d,
            radii=radii,
            depths=depths,
            tile_size=tile_size,
            tile_width=tile_width,
            tile_height=tile_height,
            sort=True,
            segmented=False,
            packed=self._config.packed,
        )
        isect_offsets = gsplat.rendering.isect_offset_encode(
            isect_ids=isect_ids,
            n_images=1,
            tile_width=tile_width,
            tile_height=tile_height,
        )

        colors = torch.zeros((1, means.shape[0], 1), device=device, dtype=dtype)
        backgrounds = torch.zeros((1, 1), device=device, dtype=dtype)
        alpha_bins = []
        for k in range(self._config.n_bins):
            opa_k = opacities * (z_cam < bins[k]).to(opacities.dtype)
            _rendered, alphas = gsplat.rendering.rasterize_to_pixels(
                means2d=means2d,
                conics=conics,
                colors=colors,
                opacities=opa_k[None, ...],
                image_width=width,
                image_height=height,
                tile_size=tile_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                backgrounds=backgrounds,
                packed=self._config.packed,
                absgrad=False,
            )
            alpha_bins.append(alphas[0, ..., 0])

        alpha_stack = torch.stack(alpha_bins, dim=0)  # (B, H, W)
        alpha_before = alpha_stack.gather(0, bin_idx.unsqueeze(0)).squeeze(0)
        return alpha_before[valid].mean()

    @staticmethod
    @torch.no_grad()
    def _camera_z(viewmat: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        num_points = means.shape[0]
        ones = torch.ones((num_points, 1), device=means.device, dtype=means.dtype)
        xyz1 = torch.cat([means, ones], dim=1)
        cam = xyz1 @ viewmat.T
        return cam[:, 2]

    @staticmethod
    def _exclude_mask(
        mask: object, shape: tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        mask_t = (
            mask.to(device=device)
            if isinstance(mask, torch.Tensor)
            else torch.as_tensor(mask, device=device)
        )
        if mask_t.dim() == 3 and mask_t.shape[0] == 1:
            mask_t = mask_t[0]
        if mask_t.shape != shape:
            raise ValueError(f"mask must have shape {shape}; got {tuple(mask_t.shape)}")
        return ~mask_t.bool()

    def _log(
        self,
        key: str,
        message: str,
        *,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        if not self._config.debug:
            return
        context: list[str] = []
        if sample is not None:
            scene = sample.get("scene")
            camera = sample.get("camera")
            timestamp = sample.get("timestamp")
            if scene is not None:
                context.append(f"scene={scene}")
            if camera is not None:
                context.append(f"camera={camera}")
            if timestamp is not None:
                context.append(f"timestamp={timestamp}")
        if step is not None:
            context.append(f"step={step}")
        if extra:
            for key_extra, value in extra.items():
                if value is None:
                    continue
                context.append(f"{key_extra}={value}")
        suffix = f" ({', '.join(context)})" if context else ""
        print(f"[FreeSpaceLoss] {message}{suffix}")

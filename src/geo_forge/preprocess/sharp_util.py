from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Final, Sequence

import gsplat
import numpy as np
import torch

from sharp.utils import color_space as color_space_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    apply_transform,
    convert_rgb_to_spherical_harmonics,
    convert_spherical_harmonics_to_rgb,
    load_ply,
)


@dataclass
class OptimizeScaleConfig:
    enabled: bool = True
    steps: int = 200
    lr: float = 1e-2
    init_scale: float = 1.0
    min_scale: float = 1e-3
    max_scale: float = 1e3
    min_depth: float = 0.1
    max_depth: float | None = 10.0
    tile_size: int = 16
    near_plane: float = 0.01
    far_plane: float = 1e10


def _flatten_gaussians(gaussians: Gaussians3D) -> Gaussians3D:
    """
    Normalize a SHARP Gaussians3D container into unbatched (N, *) tensors.

    SHARP utilities sometimes return batched Gaussians with shape (B, N, D). This
    helper flattens the batch dimension so downstream code can assume (N, D)
    (and opacities (N,) or (N, 1)).
    """
    mean_vectors = gaussians.mean_vectors
    if mean_vectors.dim() == 3:
        mean_vectors = mean_vectors.flatten(0, 1)

    singular_values = gaussians.singular_values
    if singular_values.dim() == 3:
        singular_values = singular_values.flatten(0, 1)

    quaternions = gaussians.quaternions
    if quaternions.dim() == 3:
        quaternions = quaternions.flatten(0, 1)

    colors = gaussians.colors
    if colors.dim() == 3:
        colors = colors.flatten(0, 1)

    opacities = gaussians.opacities
    if opacities.dim() == 2:
        opacities = opacities.flatten(0, 1)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def _ensure_batch_gaussians(gaussians: Gaussians3D) -> Gaussians3D:
    """
    Ensure Gaussians3D fields carry a batch dimension (B, N, ...).
    """

    def _ensure(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            return t.unsqueeze(0)
        if t.dim() == 1:
            return t.unsqueeze(0)
        return t

    return Gaussians3D(
        mean_vectors=_ensure(gaussians.mean_vectors),
        singular_values=_ensure(gaussians.singular_values),
        quaternions=_ensure(gaussians.quaternions),
        colors=_ensure(gaussians.colors),
        opacities=_ensure(gaussians.opacities),
    )


def _concat_gaussians(gaussians_list: Sequence[Gaussians3D]) -> Gaussians3D:
    """
    Concatenate multiple Gaussians3D containers along the Gaussian dimension.
    """
    if not gaussians_list:
        raise ValueError("gaussians_list must be non-empty.")

    flattened = [_flatten_gaussians(g) for g in gaussians_list]
    device = flattened[0].mean_vectors.device
    dtype = flattened[0].mean_vectors.dtype

    def _ensure(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=device, dtype=dtype)

    mean_vectors = torch.cat([_ensure(g.mean_vectors) for g in flattened], dim=0)
    singular_values = torch.cat([_ensure(g.singular_values) for g in flattened], dim=0)
    quaternions = torch.cat([_ensure(g.quaternions) for g in flattened], dim=0)
    colors = torch.cat([_ensure(g.colors) for g in flattened], dim=0)
    opacities = torch.cat([_ensure(g.opacities) for g in flattened], dim=0)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=singular_values,
        quaternions=quaternions,
        colors=colors,
        opacities=opacities,
    )


def _load_sharp_gaussians_world(meta: dict[str, object]) -> Gaussians3D | None:
    """
    Load SHARP-predicted Gaussians and lift them into world space via ``c2w``.

    Expected ``meta`` keys:
      - ``sharp_predicted_gaussians3d``: path string to a SHARP PLY (or None)
      - ``c2w``: 4x4 torch.Tensor camera-to-world transform

    Returns:
        Gaussians3D in world coordinates, or None when the path is missing.
    """
    ply_path_raw = meta.get("sharp_predicted_gaussians3d")
    if not isinstance(ply_path_raw, str) or not ply_path_raw:
        return None
    ply_path = Path(ply_path_raw)
    if not ply_path.exists():
        return None

    c2w_raw = meta.get("c2w")
    if not isinstance(c2w_raw, torch.Tensor) or c2w_raw.shape != (4, 4):
        raise ValueError("Sample metadata must include a 4x4 torch.Tensor 'c2w'.")
    c2w = c2w_raw.detach().to(dtype=torch.float32, device="cpu")

    # SHARP's load_ply assumes torch inputs in color space utilities; some PLYs may
    # surface numpy arrays. Patch once to coerce numpy inputs to torch tensors and
    # return numpy when the caller passed numpy.
    if not hasattr(color_space_utils, "_geoforge_numpy_safe"):
        _orig_robust_where = color_space_utils.robust_where

        def _robust_where_numpy_safe(
            condition: object,
            input: object,
            *args: object,
            **kwargs: object,
        ):
            was_numpy = isinstance(input, np.ndarray) or isinstance(
                condition, np.ndarray
            )
            cond_t = torch.as_tensor(condition)
            input_t = torch.as_tensor(input)
            out = _orig_robust_where(cond_t, input_t, *args, **kwargs)
            if was_numpy:
                return out.detach().cpu().numpy()
            return out

        color_space_utils.robust_where = _robust_where_numpy_safe  # type: ignore[assignment]
        color_space_utils._geoforge_numpy_safe = True  # type: ignore[attr-defined]

    gaussians, _ = load_ply(ply_path)
    gaussians = apply_transform(gaussians, c2w[:3, :])
    return _flatten_gaussians(gaussians)


def gaussians3d_to_splatsim(gaussians: Gaussians3D) -> list["Gaussian"]:
    """
    Convert a SHARP ``Gaussians3D`` to a splatsim ``list[Gaussian]``.

    Notes:
        - ``Gaussian.scale`` is stored in log-space (matching 3DGS PLY exporters).
        - ``Gaussian.opacity`` is stored as a logit (inverse-sigmoid).
        - ``Gaussian.f_dc`` follows SHARP's PLY exporter: linearRGB -> sRGB -> SH-DC.
        - Fields unused by our training demo (``normal``, ``f_rest``, ``reflection``)
          are still populated with reasonable defaults.
    """
    from splatsim import Gaussian

    xyz = gaussians.mean_vectors.flatten(0, 1).detach().to(dtype=torch.float32)
    scales_log = (
        gaussians.singular_values.flatten(0, 1)
        .detach()
        .to(dtype=torch.float32)
        .clamp_min(1e-12)
        .log()
    )
    quaternions = gaussians.quaternions.flatten(0, 1).detach().to(dtype=torch.float32)
    opacities = gaussians.opacities.flatten(0, 1).detach().to(dtype=torch.float32)
    opacity_logits = torch.logit(opacities.clamp(1e-6, 1.0 - 1e-6)).squeeze(-1)

    # SHARP predicts linearRGB, while most downstream tools expect sRGB-like values.
    colors_sh_dc = convert_rgb_to_spherical_harmonics(
        color_space_utils.linearRGB2sRGB(
            gaussians.colors.flatten(0, 1).detach().to(dtype=torch.float32)
        )
    )

    def _quat_rotate_vector(
        quats_wxyz: torch.Tensor, vectors_xyz: torch.Tensor
    ) -> torch.Tensor:
        q_vec = quats_wxyz[..., 1:4]
        q_w = quats_wxyz[..., 0:1]
        uv = torch.cross(q_vec, vectors_xyz, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vectors_xyz + 2.0 * (q_w * uv + uuv)

    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=xyz.device)
    normals = _quat_rotate_vector(quaternions, z_axis.expand_as(xyz)).cpu().numpy()

    xyz_np = xyz.cpu().numpy()
    scales_np = scales_log.cpu().numpy()
    quats_np = quaternions.cpu().numpy()
    op_np = opacity_logits.cpu().numpy()
    colors_np = colors_sh_dc.cpu().numpy()

    splat_gaussians: list[Gaussian] = []
    for idx in range(xyz_np.shape[0]):
        splat_gaussians.append(
            Gaussian(
                position=tuple(float(v) for v in xyz_np[idx]),
                normal=tuple(float(v) for v in normals[idx]),
                f_dc=tuple(float(v) for v in colors_np[idx]),
                f_rest=(),
                opacity=float(op_np[idx]),
                scale=tuple(float(v) for v in scales_np[idx]),
                rot=tuple(float(v) for v in quats_np[idx]),
                reflection=0.0,
            )
        )
    return splat_gaussians


def _render_depth_gsplat(
    *,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    intrinsics_3x3: torch.Tensor,
    c2w_4x4: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    background_depth: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Render depth/alpha with gsplat from raw tensors.

    Note: This function intentionally does not detach inputs so it can be used
    in optimization loops (e.g. optimize_scale).
    """
    viewmat = torch.inverse(c2w_4x4)[None, ...]
    Ks = intrinsics_3x3[None, ...]

    (radii, means2d, depths, conics, _) = gsplat.rendering.fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmat,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        opacities=opacities,
        packed=False,
    )

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
        packed=False,
    )
    isect_offsets = gsplat.rendering.isect_offset_encode(
        isect_ids=isect_ids,
        n_images=1,
        tile_width=tile_width,
        tile_height=tile_height,
    )

    if depths.dim() == 3 and depths.shape[-1] == 1:
        depths = depths.squeeze(-1)
    if depths.dim() != 2:
        raise ValueError(f"Unexpected depths shape from gsplat: {tuple(depths.shape)}")

    depth_colors = depths[..., None]  # (1, N, 1)
    backgrounds = torch.zeros((1, 1), device=means.device, dtype=torch.float32)
    rendered, alphas = gsplat.rendering.rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=depth_colors,
        opacities=opacities[None, ...],
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        packed=False,
        absgrad=False,
    )

    depth_weighted = rendered[0, ..., 0].to(dtype=torch.float32)
    alpha = alphas[0, ..., 0].to(dtype=torch.float32)
    background_value = (
        float(far_plane) if background_depth is None else float(background_depth)
    )
    depth = torch.where(
        alpha > 0.0,
        depth_weighted / torch.clamp(alpha, min=1e-8),
        depth_weighted.new_full((height, width), background_value),
    )
    return depth, alpha


def render_depth(
    gaussians: Gaussians3D,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    background_depth: float | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Render a metric depth image from SHARP ``Gaussians3D`` using gsplat.

    The returned depth is the alpha-normalized expected depth per pixel in the
    camera coordinate system produced by gsplat's pinhole projection.

    Args:
        gaussians: SHARP Gaussians3D container.
        intrinsics: Camera intrinsics, shape (3, 3) or (4, 4).
        c2w: Camera-to-world transform, shape (4, 4).
        width: Output image width in pixels.
        height: Output image height in pixels.
        tile_size: Tile size passed to gsplat rasterizer.
        near_plane: Near plane for projection.
        far_plane: Far plane for projection.
        background_depth: Depth used where alpha is zero; defaults to ``far_plane``.
        device: Device for rendering; defaults to ``intrinsics.device``.

    Returns:
        (depth, alpha) where both are float32 tensors of shape (H, W).
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive; got {(width, height)}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive; got {tile_size}")

    target_device = (
        torch.device(device)
        if device is not None
        else (
            intrinsics.device
            if isinstance(intrinsics, torch.Tensor)
            else torch.device("cpu")
        )
    )
    if target_device.type != "cuda":
        raise RuntimeError(
            "render_depth requires a CUDA device because gsplat's "
            "fully_fused_projection is CUDA-only in this environment. "
            "Pass device='cuda' (and ensure a working CUDA runtime) to render depth."
        )

    means = gaussians.mean_vectors.detach()
    if means.dim() == 3:
        if means.shape[0] != 1:
            raise ValueError(
                "render_depth currently supports batch size 1 for Gaussians3D; "
                f"got mean_vectors batch {means.shape[0]}"
            )
        means = means.flatten(0, 1)
    if means.dim() != 2 or means.shape[-1] != 3:
        raise ValueError(
            "gaussians.mean_vectors must have shape (N, 3) or (1, N, 3); "
            f"got {tuple(gaussians.mean_vectors.shape)}"
        )
    means = means.to(device=target_device, dtype=torch.float32)

    scales = gaussians.singular_values.detach()
    if scales.dim() == 3:
        scales = scales.flatten(0, 1)
    if scales.shape != means.shape:
        raise ValueError(
            "gaussians.singular_values must match mean_vectors shape; "
            f"got {tuple(gaussians.singular_values.shape)}"
        )
    scales = scales.to(device=target_device, dtype=torch.float32).clamp_min(1e-6)

    quats = gaussians.quaternions.detach()
    if quats.dim() == 3:
        quats = quats.flatten(0, 1)
    if quats.dim() != 2 or quats.shape != (means.shape[0], 4):
        raise ValueError(
            "gaussians.quaternions must have shape (N, 4) or (1, N, 4); "
            f"got {tuple(gaussians.quaternions.shape)}"
        )
    quats = quats.to(device=target_device, dtype=torch.float32)
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)

    opacities = gaussians.opacities.detach()
    if opacities.dim() == 3 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    if opacities.dim() == 2:
        if opacities.shape[0] != 1:
            raise ValueError(
                "render_depth currently supports batch size 1 for Gaussians3D opacities; "
                f"got {tuple(gaussians.opacities.shape)}"
            )
        opacities = opacities.flatten(0, 1)
    if opacities.dim() != 1 or opacities.shape[0] != means.shape[0]:
        raise ValueError(
            "gaussians.opacities must have shape (N,), (N, 1), (1, N), or (1, N, 1); "
            f"got {tuple(gaussians.opacities.shape)}"
        )
    opacities = opacities.to(device=target_device, dtype=torch.float32).clamp(0.0, 1.0)

    K = intrinsics.detach().to(device=target_device, dtype=torch.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(
            f"intrinsics must have shape (3, 3) or (4, 4); got {tuple(intrinsics.shape)}"
        )

    c2w_t = c2w.detach().to(device=target_device, dtype=torch.float32)
    if c2w_t.shape != (4, 4):
        raise ValueError(f"c2w must have shape (4, 4); got {tuple(c2w.shape)}")

    return _render_depth_gsplat(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        intrinsics_3x3=K,
        c2w_4x4=c2w_t,
        width=width,
        height=height,
        tile_size=tile_size,
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        background_depth=background_depth,
    )


def _build_depth_supervision_mask(
    *,
    lidar_depth: torch.Tensor,
    sky_mask: np.ndarray | torch.Tensor | None,
    movable_object_mask: np.ndarray | torch.Tensor | None,
    max_depth: float | None,
    mask_upper_half: bool,
) -> torch.Tensor:
    """
    Build the boolean supervision mask for depth optimization.

    Valid pixels are those where LiDAR depth is finite, within the optional
    max depth threshold, and NOT inside excluded regions (sky / movable objects).
    """
    if lidar_depth.dim() != 2:
        raise ValueError(
            f"lidar_depth must have shape (H, W); got {tuple(lidar_depth.shape)}"
        )
    height, width = int(lidar_depth.shape[0]), int(lidar_depth.shape[1])
    shape_hw = (height, width)

    def _as_bool_mask(mask_in: np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if mask_in is None:
            return torch.zeros(shape_hw, dtype=torch.bool, device=lidar_depth.device)
        mask_t = (
            torch.from_numpy(np.asarray(mask_in))
            if isinstance(mask_in, np.ndarray)
            else mask_in
        )
        if mask_t.dim() == 3 and mask_t.shape[0] == 1:
            mask_t = mask_t[0]
        if mask_t.dim() != 2:
            raise ValueError(
                f"Mask must have shape (H, W) (or (1, H, W)); got {tuple(mask_t.shape)}"
            )
        if tuple(int(v) for v in mask_t.shape) != shape_hw:
            raise ValueError(
                f"Mask shape must match lidar_depth shape {shape_hw}; got {tuple(mask_t.shape)}"
            )
        return mask_t.to(device=lidar_depth.device).bool()

    exclude_sky = _as_bool_mask(sky_mask)
    exclude_obj = _as_bool_mask(movable_object_mask)

    valid = torch.isfinite(lidar_depth) & (~exclude_sky) & (~exclude_obj)
    if max_depth is not None:
        valid &= lidar_depth <= float(max_depth)
    if mask_upper_half:
        valid[: height // 2] = False
    return valid


def filter_gaussians_by_skymask(
    gaussians: Gaussians3D,
    sky_mask: np.ndarray | torch.Tensor | None,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    drop_out_of_view: bool = False,
) -> Gaussians3D:
    """
    Remove Gaussians whose projected mean falls inside the sky mask.

    Projection:
      - transform world points into camera frame using ``w2c = inverse(c2w)``
      - apply pinhole projection with ``K``: ``u = fx*x/z + cx``, ``v = fy*y/z + cy``

    Args:
        gaussians: SHARP Gaussians3D (batch size 1 supported).
        sky_mask: Sky mask (H, W). True indicates sky pixels to exclude. If None,
            no sky-based filtering is performed (but out-of-view dropping can still
            be enabled if ``image_width``/``image_height`` are provided).
        intrinsics: Camera intrinsics (3, 3) or (4, 4).
        c2w: Camera-to-world transform (4, 4).
        image_width: Image width used for out-of-view filtering when ``sky_mask`` is None.
        image_height: Image height used for out-of-view filtering when ``sky_mask`` is None.
        drop_out_of_view: If True, also drop Gaussians that project outside the image
            bounds or behind the camera (z <= 0).

    Returns:
        Filtered Gaussians3D (batch size preserved).
    """
    means = gaussians.mean_vectors
    if means.dim() == 3:
        if means.shape[0] != 1:
            raise ValueError(
                "filter_gaussians_by_skymask currently supports batch size 1; "
                f"got mean_vectors batch {means.shape[0]}"
            )
        means = means[0]
    if means.dim() != 2 or means.shape[-1] != 3:
        raise ValueError(
            "gaussians.mean_vectors must have shape (N, 3) or (1, N, 3); "
            f"got {tuple(gaussians.mean_vectors.shape)}"
        )

    K = intrinsics.to(dtype=torch.float32)
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(
            f"intrinsics must have shape (3, 3) or (4, 4); got {tuple(intrinsics.shape)}"
        )
    c2w_t = c2w.to(dtype=torch.float32)
    if c2w_t.shape != (4, 4):
        raise ValueError(f"c2w must have shape (4, 4); got {tuple(c2w.shape)}")

    device = means.device
    K = K.to(device=device)
    w2c = torch.inverse(c2w_t.to(device=device))

    ones = torch.ones((means.shape[0], 1), dtype=torch.float32, device=device)
    means_h = torch.cat([means.to(dtype=torch.float32), ones], dim=-1)  # (N, 4)
    cam = means_h @ w2c.T
    x = cam[:, 0]
    y = cam[:, 1]
    z = cam[:, 2]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z_safe = torch.where(z.abs() < 1e-12, z.new_full((), 1e-12), z)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy

    if sky_mask is None:
        if drop_out_of_view and (image_width is None or image_height is None):
            raise ValueError(
                "When sky_mask is None and drop_out_of_view is True, "
                "image_width and image_height must be provided."
            )
        width = int(image_width) if image_width is not None else 0
        height = int(image_height) if image_height is not None else 0
        sky_mask_t = None
    else:
        sky_mask_t = (
            torch.from_numpy(np.asarray(sky_mask))
            if isinstance(sky_mask, np.ndarray)
            else sky_mask
        )
        if sky_mask_t.dim() != 2:
            raise ValueError(
                f"sky_mask must have shape (H, W); got {tuple(sky_mask_t.shape)}"
            )
        sky_mask_t = sky_mask_t.to(device=device).bool()
        height, width = int(sky_mask_t.shape[0]), int(sky_mask_t.shape[1])

    inside = (z > 0.0) & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    u_idx = torch.round(u).to(dtype=torch.int64).clamp(0, max(0, width - 1))
    v_idx = torch.round(v).to(dtype=torch.int64).clamp(0, max(0, height - 1))

    in_sky = torch.zeros((means.shape[0],), dtype=torch.bool, device=device)
    if sky_mask_t is not None and inside.any():
        in_sky[inside] = sky_mask_t[v_idx[inside], u_idx[inside]]

    keep = ~in_sky
    if drop_out_of_view:
        keep &= inside

    def _filter_field(field: torch.Tensor, last_dim: int) -> torch.Tensor:
        if field.dim() == 3:
            field = field[0]
        if field.dim() != 2 or field.shape[-1] != last_dim:
            raise ValueError(
                f"Unexpected Gaussians3D field shape {tuple(field.shape)}; "
                f"expected (N, {last_dim})"
            )
        return field[keep][None, ...]

    def _filter_1d(field: torch.Tensor) -> torch.Tensor:
        if field.dim() == 2:
            field = field[0]
        if field.dim() != 1:
            raise ValueError(f"Unexpected Gaussians3D field shape {tuple(field.shape)}")
        return field[keep][None, ...]

    return Gaussians3D(
        mean_vectors=_filter_field(gaussians.mean_vectors, 3),
        singular_values=_filter_field(gaussians.singular_values, 3),
        quaternions=_filter_field(gaussians.quaternions, 4),
        colors=_filter_field(gaussians.colors, 3),
        opacities=_filter_1d(gaussians.opacities),
    )


def optimize_scale(
    gaussians: Gaussians3D,
    lidar_depth: np.ndarray | torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    *,
    sky_mask: np.ndarray | torch.Tensor | None = None,
    movable_object_mask: np.ndarray | torch.Tensor | None = None,
    max_depth: float | None = 10.0,
    mask_upper_half: bool = True,
    steps: int = 200,
    lr: float = 1e-2,
    init_scale: float = 1.0,
    min_scale: float = 1e-3,
    max_scale: float = 1e3,
    tile_size: int = 16,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    device: torch.device | str = "cuda",
    verbose: bool = False,
) -> tuple[Gaussians3D, float, list[float]]:
    """
    Optimize a global metric scale factor by matching rendered depth to LiDAR depth.

    This routine assumes SHARP Gaussians may be up-to-scale and finds a scalar
    ``s`` such that rendering depth from:

      - ``mean_vectors * s`` and ``singular_values * s``

    minimizes the masked L1 depth error against a LiDAR-projected depth image.

    Masking:
      - A binary mask is built from the LiDAR depth image as ``isfinite(depth)``.
      - Pixels inside ``sky_mask`` or ``movable_object_mask`` are excluded.
      - Pixels deeper than ``max_depth`` are excluded.
      - When ``mask_upper_half`` is True, pixels in the upper half of the image
        are excluded.
      - The loss is computed only on valid (masked) pixels.

    Args:
        gaussians: SHARP Gaussians3D (batch size 1 supported).
        lidar_depth: LiDAR-projected depth image (H, W). Pixels without LiDAR should
            be NaN (recommended) so they are excluded by the mask.
        intrinsics: Camera intrinsics (3, 3) or (4, 4).
        c2w: Camera-to-world transform (4, 4).
        sky_mask: Optional sky mask (H, W). True means exclude the pixel.
        movable_object_mask: Optional movable object mask (H, W). True means exclude.
        max_depth: Optional maximum depth (meters) used for supervision masking.
        mask_upper_half: If True, exclude the image upper half from supervision.
        steps: Optimization steps.
        lr: Adam learning rate (in log-scale space).
        init_scale: Initial scale factor (>0).
        min_scale: Minimum allowed scale factor (>0).
        max_scale: Maximum allowed scale factor (>min_scale).
        tile_size: Tile size passed to gsplat rasterizer.
        near_plane: Near plane for projection.
        far_plane: Far plane for projection.
        device: CUDA device for rendering/optimization.
        verbose: If True, prints loss/scale occasionally.

    Returns:
        (scaled_gaussians, scale, loss_history)
    """
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise RuntimeError(
            "optimize_scale requires a CUDA device because gsplat's "
            "fully_fused_projection is CUDA-only in this environment."
        )

    lidar_depth_t = (
        torch.from_numpy(np.asarray(lidar_depth))
        if isinstance(lidar_depth, np.ndarray)
        else lidar_depth
    )
    if lidar_depth_t.dim() != 2:
        raise ValueError(
            f"lidar_depth must have shape (H, W); got {tuple(lidar_depth_t.shape)}"
        )
    lidar_depth_t = lidar_depth_t.to(device=target_device, dtype=torch.float32)
    height, width = int(lidar_depth_t.shape[0]), int(lidar_depth_t.shape[1])

    gaussians = filter_gaussians_by_skymask(
        gaussians=gaussians,
        sky_mask=sky_mask,
        intrinsics=intrinsics,
        c2w=c2w,
        image_width=width,
        image_height=height,
        drop_out_of_view=True,
    )

    mask = _build_depth_supervision_mask(
        lidar_depth=lidar_depth_t,
        sky_mask=sky_mask,
        movable_object_mask=movable_object_mask,
        max_depth=max_depth,
        mask_upper_half=bool(mask_upper_half),
    )
    mask_f = mask.to(dtype=torch.float32)
    valid_count = int(mask.sum().item())
    if valid_count == 0:
        raise ValueError(
            "No valid pixels to supervise with after applying LiDAR finite mask "
            "and excluding sky/movable object regions."
        )
    lidar_depth_filled = torch.nan_to_num(lidar_depth_t, nan=0.0)

    means0 = gaussians.mean_vectors
    if means0.dim() == 3:
        if means0.shape[0] != 1:
            raise ValueError(
                "optimize_scale currently supports batch size 1 for Gaussians3D; "
                f"got mean_vectors batch {means0.shape[0]}"
            )
        means0 = means0.flatten(0, 1)
    if means0.dim() != 2 or means0.shape[-1] != 3:
        raise ValueError(
            "gaussians.mean_vectors must have shape (N, 3) or (1, N, 3); "
            f"got {tuple(gaussians.mean_vectors.shape)}"
        )
    means0 = means0.to(device=target_device, dtype=torch.float32).detach()

    scales0 = gaussians.singular_values
    if scales0.dim() == 3:
        scales0 = scales0.flatten(0, 1)
    if scales0.shape != means0.shape:
        raise ValueError(
            "gaussians.singular_values must match mean_vectors shape; "
            f"got {tuple(gaussians.singular_values.shape)}"
        )
    scales0 = (
        scales0.to(device=target_device, dtype=torch.float32).detach().clamp_min(1e-6)
    )

    quats = gaussians.quaternions
    if quats.dim() == 3:
        quats = quats.flatten(0, 1)
    if quats.dim() != 2 or quats.shape != (means0.shape[0], 4):
        raise ValueError(
            "gaussians.quaternions must have shape (N, 4) or (1, N, 4); "
            f"got {tuple(gaussians.quaternions.shape)}"
        )
    quats = quats.to(device=target_device, dtype=torch.float32).detach()
    quats = quats / torch.clamp(quats.norm(dim=-1, keepdim=True), min=1e-12)

    opacities = gaussians.opacities
    if opacities.dim() == 3 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    if opacities.dim() == 2:
        if opacities.shape[0] != 1:
            raise ValueError(
                "optimize_scale currently supports batch size 1 for Gaussians3D opacities; "
                f"got {tuple(gaussians.opacities.shape)}"
            )
        opacities = opacities.flatten(0, 1)
    if opacities.dim() != 1 or opacities.shape[0] != means0.shape[0]:
        raise ValueError(
            "gaussians.opacities must have shape (N,), (N, 1), (1, N), or (1, N, 1); "
            f"got {tuple(gaussians.opacities.shape)}"
        )
    opacities = (
        opacities.to(device=target_device, dtype=torch.float32).detach().clamp(0.0, 1.0)
    )

    K = intrinsics.to(device=target_device, dtype=torch.float32).detach()
    if K.shape == (4, 4):
        K = K[:3, :3]
    if K.shape != (3, 3):
        raise ValueError(
            f"intrinsics must have shape (3, 3) or (4, 4); got {tuple(intrinsics.shape)}"
        )

    c2w_t = c2w.to(device=target_device, dtype=torch.float32).detach()
    if c2w_t.shape != (4, 4):
        raise ValueError(f"c2w must have shape (4, 4); got {tuple(c2w.shape)}")

    init_scale = float(init_scale)
    if init_scale <= 0.0:
        raise ValueError(f"init_scale must be > 0; got {init_scale}")
    min_scale = float(min_scale)
    max_scale = float(max_scale)
    if min_scale <= 0.0:
        raise ValueError(f"min_scale must be > 0; got {min_scale}")
    if max_scale <= min_scale:
        raise ValueError(f"max_scale must be > min_scale; got {(min_scale, max_scale)}")
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(init_scale), device=target_device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([log_scale], lr=float(lr))
    loss_history: list[float] = []
    log_min = float(math.log(min_scale))
    log_max = float(math.log(max_scale))

    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        log_scale.data.clamp_(log_min, log_max)
        scale = torch.exp(log_scale)
        means = means0 * scale
        scales = (scales0 * scale).clamp_min(1e-6)

        depth_pred, _alpha = _render_depth_gsplat(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            intrinsics_3x3=K,
            c2w_4x4=c2w_t,
            width=width,
            height=height,
            tile_size=tile_size,
            near_plane=float(near_plane),
            far_plane=float(far_plane),
            background_depth=float("nan"),
        )
        depth_pred_filled = torch.nan_to_num(depth_pred, nan=0.0)

        loss = (torch.abs(depth_pred_filled - lidar_depth_filled) * mask_f).sum() / (
            mask_f.sum() + 1e-8
        )
        loss.backward()
        optimizer.step()

        log_scale.data.clamp_(log_min, log_max)
        loss_history.append(float(loss.detach().cpu()))
        if verbose and (step == 0 or (step + 1) % 25 == 0 or step + 1 == steps):
            print(
                f"[optimize_scale step={step + 1:04d}] "
                f"loss={loss_history[-1]:.6f} scale={float(scale.detach().cpu()):.6f} "
                f"valid_pixels={valid_count}"
            )

    scale_final = float(torch.exp(log_scale.detach()).cpu())
    scaled_gaussians = Gaussians3D(
        mean_vectors=gaussians.mean_vectors * scale_final,
        singular_values=gaussians.singular_values * scale_final,
        quaternions=gaussians.quaternions,
        colors=gaussians.colors,
        opacities=gaussians.opacities,
    )
    return scaled_gaussians, scale_final, loss_history


def _build_default_intrinsics(f_px: float, width: int, height: int) -> torch.Tensor:
    """Create a 3x3 pinhole intrinsics matrix matching SHARP's PLY exporter."""
    return torch.tensor(
        [
            [f_px, 0.0, width * 0.5],
            [0.0, f_px, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _load_gaussians_from_sharp_ply(
    path: Path,
) -> tuple[Gaussians3D, torch.Tensor, torch.Tensor, int, int]:
    """
    Load Gaussians3D + camera metadata from a SHARP-exported PLY.

    This re-implements SHARP's ``load_ply`` with a small fix: upstream currently
    mixes numpy arrays and torch tensors when decoding colors, which breaks on
    recent PyTorch versions.

    Returns:
        (gaussians, intrinsics, c2w, width, height)
    """
    from sharp.utils.gaussians import PlyData

    plydata = PlyData.read(path)
    vertices = next(filter(lambda x: x.name == "vertex", plydata.elements))

    required_props = ["x", "y", "z", "opacity"]
    required_props += [f"f_dc_{i}" for i in range(3)]
    required_props += [f"scale_{i}" for i in range(3)]
    required_props += [f"rot_{i}" for i in range(4)]
    for prop in required_props:
        if prop not in vertices:
            raise KeyError(
                f"Incompatible ply file: property {prop} not found in ply elements."
            )

    means_np = np.stack(
        (
            np.asarray(vertices["x"], dtype=np.float32),
            np.asarray(vertices["y"], dtype=np.float32),
            np.asarray(vertices["z"], dtype=np.float32),
        ),
        axis=1,
    )
    scale_logits_np = np.stack(
        (
            np.asarray(vertices["scale_0"], dtype=np.float32),
            np.asarray(vertices["scale_1"], dtype=np.float32),
            np.asarray(vertices["scale_2"], dtype=np.float32),
        ),
        axis=1,
    )
    quats_np = np.stack(
        (
            np.asarray(vertices["rot_0"], dtype=np.float32),
            np.asarray(vertices["rot_1"], dtype=np.float32),
            np.asarray(vertices["rot_2"], dtype=np.float32),
            np.asarray(vertices["rot_3"], dtype=np.float32),
        ),
        axis=1,
    )
    sh0_np = np.stack(
        (
            np.asarray(vertices["f_dc_0"], dtype=np.float32),
            np.asarray(vertices["f_dc_1"], dtype=np.float32),
            np.asarray(vertices["f_dc_2"], dtype=np.float32),
        ),
        axis=1,
    )
    opacity_logits_np = np.asarray(vertices["opacity"], dtype=np.float32)[..., None]

    supplement_elements = [
        element for element in plydata.elements if element.name != "vertex"
    ]
    supplement_data: dict[str, np.ndarray] = {}
    supplement_keys = ["extrinsic", "intrinsic", "color_space", "image_size"]
    for element in supplement_elements:
        for key in supplement_keys:
            if key not in supplement_data and key in element:
                supplement_data[key] = np.asarray(element[key])

    # Intrinsics + image size.
    if "intrinsic" in supplement_data:
        intrinsic_data = supplement_data["intrinsic"]
        if "image_size" not in supplement_data:
            # Legacy: [fx, fy, width, height]
            if len(intrinsic_data) != 4:
                raise ValueError(
                    "Expected legacy intrinsics with len=4 containing image size, "
                    f"but received len={len(intrinsic_data)}"
                )
            fx = float(intrinsic_data[0])
            fy = float(intrinsic_data[1])
            width = int(intrinsic_data[2])
            height = int(intrinsic_data[3])
            cx = width * 0.5
            cy = height * 0.5
            intrinsics = torch.tensor(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        else:
            if len(intrinsic_data) != 9:
                raise ValueError(
                    f"Expected 9 elements in intrinsics, but received {len(intrinsic_data)}."
                )
            intrinsics = torch.tensor(
                intrinsic_data.reshape((3, 3)), dtype=torch.float32
            )
            image_size_data = supplement_data["image_size"]
            width = int(image_size_data[0])
            height = int(image_size_data[1])
    else:
        width, height = 640, 480
        intrinsics = _build_default_intrinsics(f_px=512.0, width=width, height=height)

    # Extrinsics.
    extrinsics_data = supplement_data.get("extrinsic", np.eye(4, dtype=np.float32))
    extrinsics_data = np.asarray(extrinsics_data, dtype=np.float32).reshape(-1)
    c2w = np.eye(4, dtype=np.float32)
    if extrinsics_data.size == 12:
        c2w[:3] = extrinsics_data.reshape((3, 4))
        c2w[:3, :3] = c2w[:3, :3].copy().T
    elif extrinsics_data.size == 16:
        c2w[:] = extrinsics_data.reshape((4, 4))
    else:
        raise ValueError(
            f"Unrecognized extrinsics matrix shape {extrinsics_data.size} in {path}"
        )

    # Decode colorspace and convert SH0 -> RGB.
    color_space_index = supplement_data.get(
        "color_space", np.array([1], dtype=np.uint8)
    )
    if isinstance(color_space_index, np.ndarray):
        color_space_index_val = int(color_space_index.reshape(-1)[0])
    else:
        color_space_index_val = int(color_space_index)
    color_space = color_space_utils.decode_color_space(color_space_index_val)

    sh0 = torch.from_numpy(sh0_np).to(dtype=torch.float32)
    colors = convert_spherical_harmonics_to_rgb(sh0)
    if color_space == "sRGB":
        colors = color_space_utils.sRGB2linearRGB(colors)

    gaussians = Gaussians3D(
        mean_vectors=torch.from_numpy(means_np).view(1, -1, 3).to(dtype=torch.float32),
        quaternions=torch.from_numpy(quats_np).view(1, -1, 4).to(dtype=torch.float32),
        singular_values=torch.exp(
            torch.from_numpy(scale_logits_np).view(1, -1, 3).to(dtype=torch.float32)
        ),
        opacities=torch.sigmoid(
            torch.from_numpy(opacity_logits_np).view(1, -1).to(dtype=torch.float32)
        ),
        colors=colors.view(1, -1, 3).to(dtype=torch.float32),
    )
    return (
        gaussians,
        intrinsics,
        torch.from_numpy(c2w).to(dtype=torch.float32),
        width,
        height,
    )


def _point_depths_camera_z(gaussians: Gaussians3D, c2w: torch.Tensor) -> torch.Tensor:
    """
    Compute per-Gaussian camera-space z (forward) depth.

    This is a CPU-friendly fallback when CUDA rasterization is unavailable.
    """
    means = gaussians.mean_vectors.detach()
    if means.dim() == 3:
        if means.shape[0] != 1:
            raise ValueError(f"Expected batch size 1; got {tuple(means.shape)}")
        means = means[0]
    if means.dim() != 2 or means.shape[-1] != 3:
        raise ValueError(
            "gaussians.mean_vectors must have shape (N, 3) or (1, N, 3); "
            f"got {tuple(gaussians.mean_vectors.shape)}"
        )
    if c2w.shape != (4, 4):
        raise ValueError(f"c2w must have shape (4, 4); got {tuple(c2w.shape)}")

    w2c = torch.inverse(c2w.to(dtype=torch.float32))
    ones = torch.ones((means.shape[0], 1), dtype=torch.float32, device=means.device)
    means_h = torch.cat([means.to(dtype=torch.float32), ones], dim=-1)  # (N, 4)
    cam = means_h @ w2c.T
    return cam[:, 2]


def _assert_cuda_usable() -> None:
    """Fail fast with a clear error if CUDA is not actually usable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to render depth with gsplat, but torch.cuda.is_available() "
            "is False. Install a CUDA-enabled PyTorch build and ensure a working driver."
        )
    try:
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on runtime/driver
        raise RuntimeError(
            "CUDA appears available but is not usable (driver/runtime issue). "
            "Fix your CUDA runtime, then rerun."
        ) from exc


def transform_gaussians3d(
    *,
    gaussians_camera: Gaussians3D,
    camera_c2w: torch.Tensor,
) -> Gaussians3D:
    """
    Transform camera-centric Gaussians3D into world coordinates.

    This is useful when SHARP (or other pipelines) produce Gaussians3D in a gsplat
    coordinate system where the camera pose is the origin. Given the camera pose
    in a world coordinate system (also gsplat / right-handed), this function:

      - transforms Gaussian means: ``x_world = R * x_cam + t``
      - composes orientations: ``R_world = R * R_gaussian``
      - leaves singular values, colors, and opacities unchanged

    Args:
        gaussians_camera: Gaussians3D defined in the camera frame. Batch size 1 is
            supported (shape (1, N, ...)).
        camera_c2w: Camera-to-world transform, shape (4, 4).

    Returns:
        World-coordinate Gaussians3D.
    """

    def _rotmat_to_quat_wxyz(rot_3x3: torch.Tensor) -> torch.Tensor:
        if rot_3x3.shape != (3, 3):
            raise ValueError(
                f"Expected rot_3x3 shape (3, 3); got {tuple(rot_3x3.shape)}"
            )
        r = rot_3x3.to(dtype=torch.float32)
        m00, m01, m02 = r[0, 0], r[0, 1], r[0, 2]
        m10, m11, m12 = r[1, 0], r[1, 1], r[1, 2]
        m20, m21, m22 = r[2, 0], r[2, 1], r[2, 2]

        trace = m00 + m11 + m22
        if float(trace) > 0.0:
            s = torch.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif float(m00) > float(m11) and float(m00) > float(m22):
            s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif float(m11) > float(m22):
            s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s
        quat = torch.stack([qw, qx, qy, qz], dim=0)
        return quat / torch.clamp(quat.norm(), min=1e-12)

    def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        if q1.shape[-1] != 4 or q2.shape[-1] != 4:
            raise ValueError(
                f"Expected quaternions with last dim 4; got {tuple(q1.shape)}, {tuple(q2.shape)}"
            )
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    if camera_c2w.shape != (4, 4):
        raise ValueError(
            f"camera_c2w must have shape (4, 4); got {tuple(camera_c2w.shape)}"
        )

    means = gaussians_camera.mean_vectors
    if means.dim() == 3:
        if means.shape[0] != 1:
            raise ValueError(
                "transform_gaussians3d supports batch size 1; "
                f"got {tuple(means.shape)}"
            )
        means = means[0]
    if means.dim() != 2 or means.shape[-1] != 3:
        raise ValueError(
            "gaussians_camera.mean_vectors must have shape (N, 3) or (1, N, 3); "
            f"got {tuple(gaussians_camera.mean_vectors.shape)}"
        )

    device = means.device
    c2w = camera_c2w.to(device=device, dtype=torch.float32)
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]

    means_world = (means.to(dtype=torch.float32) @ rot.T) + trans[None, :]

    quats = gaussians_camera.quaternions
    if quats.dim() == 3:
        if quats.shape[0] != 1:
            raise ValueError(
                "transform_gaussians3d supports batch size 1; "
                f"got {tuple(quats.shape)}"
            )
        quats = quats[0]
    if quats.dim() != 2 or quats.shape != (means.shape[0], 4):
        raise ValueError(
            "gaussians_camera.quaternions must have shape (N, 4) or (1, N, 4); "
            f"got {tuple(gaussians_camera.quaternions.shape)}"
        )

    q_cam = _rotmat_to_quat_wxyz(rot).to(device=device, dtype=torch.float32)
    q_cam = q_cam.expand(means.shape[0], 4)
    quats_world = _quat_mul_wxyz(q_cam, quats.to(dtype=torch.float32))
    quats_world = quats_world / torch.clamp(
        quats_world.norm(dim=-1, keepdim=True), min=1e-12
    )

    transformed = Gaussians3D(
        mean_vectors=means_world[None, ...],
        singular_values=gaussians_camera.singular_values.to(device=device),
        quaternions=quats_world[None, ...],
        colors=gaussians_camera.colors.to(device=device),
        opacities=gaussians_camera.opacities.to(device=device),
    )
    return transformed


def merge_gaussians3d(*, base: Gaussians3D, to_add: Gaussians3D) -> Gaussians3D:
    """
    Merge two Gaussians3D containers by concatenating along the Gaussian dimension.

    Both inputs must have batch size 1 and compatible feature dimensions.
    """

    device = base.mean_vectors.device

    def _cat_field(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dim() != 3 or b.dim() != 3 or a.shape[0] != 1 or b.shape[0] != 1:
            raise ValueError(
                "Expected Gaussians3D fields with shape (1, N, D); "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(
                f"Last dimension must match; got {a.shape[-1]} and {b.shape[-1]}"
            )
        return torch.cat([a.to(device=device), b.to(device=device)], dim=1)

    def _cat_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != 1 or b.shape[0] != 1:
            raise ValueError(
                "Expected Gaussians3D opacities with shape (1, N); "
                f"got {tuple(a.shape)} and {tuple(b.shape)}"
            )
        return torch.cat([a.to(device=device), b.to(device=device)], dim=1)

    return Gaussians3D(
        mean_vectors=_cat_field(base.mean_vectors, to_add.mean_vectors),
        singular_values=_cat_field(base.singular_values, to_add.singular_values),
        quaternions=_cat_field(base.quaternions, to_add.quaternions),
        colors=_cat_field(base.colors, to_add.colors),
        opacities=_cat_1d(base.opacities, to_add.opacities),
    )


__all__ = [
    "OptimizeScaleConfig",
    "gaussians3d_to_splatsim",
    "render_depth",
    "optimize_scale",
    "filter_gaussians_by_skymask",
    "transform_gaussians3d",
    "merge_gaussians3d",
]


if __name__ == "__main__":
    pass

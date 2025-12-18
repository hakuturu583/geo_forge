from __future__ import annotations

import math

import gsplat
import torch

from sharp.utils import color_space as color_space_utils
from sharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics


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
    background_value = (
        float(far_plane) if background_depth is None else float(background_depth)
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

    viewmat = torch.inverse(c2w_t)[None, ...]
    Ks = K[None, ...]

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
    backgrounds = torch.zeros((1, 1), device=target_device, dtype=torch.float32)
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

    # rendered/alphas are [1, H, W, 1]
    depth_weighted = rendered[0, ..., 0].to(dtype=torch.float32)
    alpha = alphas[0, ..., 0].to(dtype=torch.float32)
    depth = torch.where(
        alpha > 0.0,
        depth_weighted / torch.clamp(alpha, min=1e-8),
        depth_weighted.new_full((height, width), background_value),
    )
    return depth, alpha


__all__ = ["gaussians3d_to_splatsim", "render_depth"]

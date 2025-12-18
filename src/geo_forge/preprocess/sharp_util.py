from __future__ import annotations

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

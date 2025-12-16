from __future__ import annotations

import math

import gsplat
import torch


class GaussianSplattingModel(torch.nn.Module):
    """Minimal Gaussian parameter container with a gsplat render helper."""

    def __init__(self, num_gaussians: int, device: torch.device) -> None:
        super().__init__()
        self.means = torch.nn.Parameter(
            torch.randn(num_gaussians, 3, device=device) * 0.5
        )
        self.scales = torch.nn.Parameter(
            torch.full((num_gaussians, 3), 0.1, device=device)
        )
        self.rotations = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1)
        )
        self.opacities = torch.nn.Parameter(
            torch.full((num_gaussians, 1), 0.5, device=device)
        )
        self.colors = torch.nn.Parameter(torch.rand(num_gaussians, 3, device=device))

    def render(
        self,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        width: int,
        height: int,
        background: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Render a single view using gsplat primitives.

        gsplat expects world-to-camera view matrices; the NuScenes poses are camera-to-world,
        so we invert before rendering.
        """
        device = self.means.device
        viewmat = torch.inverse(c2w)[None, ...]  # (C=1, 4, 4)
        Ks = intrinsics[None, ...]  # (C=1, 3, 3)
        bg = (
            background.to(device)
            if background is not None
            else torch.zeros(3, device=device)
        ).view(1, 3)

        (
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = gsplat.rendering.fully_fused_projection(
            means=self.means,
            covars=None,
            quats=self.rotations,
            scales=self.scales,
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            opacities=self.opacities.squeeze(-1),
        )

        tile_size = 16
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

        render_out, _ = gsplat.rendering.rasterize_to_pixels(
            means2d=means2d,
            conics=conics,
            colors=self.colors[None, ...],
            opacities=self.opacities.squeeze(-1)[None, ...],
            image_width=width,
            image_height=height,
            tile_size=tile_size,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            backgrounds=bg,
            packed=False,
            absgrad=False,
        )
        # render_out: [B, C, H, W, 3]; collapse batch/cam dims.
        if render_out.dim() == 5:
            render_out = render_out.permute(0, 1, 4, 2, 3)
            render_out = render_out[0, 0]
        elif render_out.dim() == 4:
            render_out = render_out.permute(0, 3, 1, 2)[0]
        return render_out

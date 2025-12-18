import unittest

import torch
from sharp.utils.gaussians import Gaussians3D

from geo_forge.preprocess.sharp_util import render_depth


class TestSharpRenderDepth(unittest.TestCase):
    def test_render_depth_single_gaussian_center(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("gsplat depth rendering requires CUDA in this environment.")

        width = 64
        height = 48
        fx = fy = 100.0
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0

        intrinsics = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        c2w = torch.eye(4, dtype=torch.float32)

        means = torch.tensor([[[0.0, 0.0, 2.0]]], dtype=torch.float32)
        singular_values = torch.tensor([[[0.02, 0.02, 0.02]]], dtype=torch.float32)
        quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
        colors = torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float32)
        opacities = torch.tensor([[1.0]], dtype=torch.float32)

        gaussians = Gaussians3D(
            mean_vectors=means,
            singular_values=singular_values,
            quaternions=quats,
            colors=colors,
            opacities=opacities,
        )

        depth, alpha = render_depth(
            gaussians=gaussians,
            intrinsics=intrinsics,
            c2w=c2w,
            width=width,
            height=height,
            device="cuda",
            background_depth=1000.0,
        )

        self.assertEqual(tuple(depth.shape), (height, width))
        self.assertEqual(tuple(alpha.shape), (height, width))
        self.assertTrue(depth.dtype == torch.float32)
        self.assertTrue(alpha.dtype == torch.float32)

        center_x = int(round(cx))
        center_y = int(round(cy))
        center_alpha = float(alpha[center_y, center_x])
        center_depth = float(depth[center_y, center_x])

        self.assertGreater(center_alpha, 1e-3)
        self.assertGreater(center_depth, 0.0)
        self.assertLess(abs(center_depth - 2.0), 0.75)


if __name__ == "__main__":
    unittest.main()

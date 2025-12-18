import unittest

import numpy as np
import torch
from sharp.utils.gaussians import Gaussians3D

from geo_forge.preprocess.sharp_util import optimize_scale


class TestSharpOptimizeScale(unittest.TestCase):
    def test_optimize_scale_requires_cuda(self) -> None:
        gaussians = Gaussians3D(
            mean_vectors=torch.zeros((1, 1, 3), dtype=torch.float32),
            singular_values=torch.ones((1, 1, 3), dtype=torch.float32) * 0.01,
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.zeros((1, 1, 3), dtype=torch.float32),
            opacities=torch.ones((1, 1), dtype=torch.float32),
        )
        lidar_depth = np.full((8, 8), np.nan, dtype=np.float32)
        lidar_depth[4, 4] = 2.0
        K = torch.tensor([[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
        c2w = torch.eye(4, dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            optimize_scale(
                gaussians=gaussians,
                lidar_depth=lidar_depth,
                intrinsics=K,
                c2w=c2w,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch
from sharp.utils.gaussians import Gaussians3D

from geo_forge.preprocess.sharp_util import filter_gaussians_by_skymask


class TestSharpFilterGaussiansBySkyMask(unittest.TestCase):
    def test_filters_projected_means_in_sky(self) -> None:
        width = 5
        height = 5
        fx = fy = 10.0
        cx = cy = 2.0
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        c2w = torch.eye(4, dtype=torch.float32)

        # Mean0 projects to (2,2) and should be removed.
        # Mean1 projects to (3,2) and should be kept.
        means = torch.tensor([[[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]]], dtype=torch.float32)
        singular_values = torch.ones((1, 2, 3), dtype=torch.float32) * 0.01
        quats = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
        )
        colors = torch.zeros((1, 2, 3), dtype=torch.float32)
        opacities = torch.ones((1, 2), dtype=torch.float32)
        gaussians = Gaussians3D(
            mean_vectors=means,
            singular_values=singular_values,
            quaternions=quats,
            colors=colors,
            opacities=opacities,
        )

        sky_mask = np.zeros((height, width), dtype=bool)
        sky_mask[2, 2] = True

        filtered = filter_gaussians_by_skymask(
            gaussians=gaussians, sky_mask=sky_mask, intrinsics=K, c2w=c2w
        )

        self.assertEqual(tuple(filtered.mean_vectors.shape), (1, 1, 3))
        torch.testing.assert_close(
            filtered.mean_vectors[0, 0], torch.tensor([0.2, 0.0, 2.0])
        )


if __name__ == "__main__":
    unittest.main()

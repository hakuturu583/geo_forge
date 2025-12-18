import unittest

import numpy as np
import torch

from geo_forge.preprocess.sharp_util import _build_depth_supervision_mask


class TestSharpDepthSupervisionMask(unittest.TestCase):
    def test_excludes_sky_and_movable_objects(self) -> None:
        lidar_depth = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [4.0, 5.0, float("nan")],
            ],
            dtype=torch.float32,
        )
        sky = np.array(
            [
                [False, False, True],
                [False, False, False],
            ],
            dtype=bool,
        )
        obj = torch.tensor(
            [
                [False, False, False],
                [False, True, False],
            ],
            dtype=torch.bool,
        )

        mask = _build_depth_supervision_mask(
            lidar_depth=lidar_depth, sky_mask=sky, movable_object_mask=obj
        )

        expected = torch.tensor(
            [
                [True, False, False],  # third excluded by sky
                [True, False, False],  # middle excluded by obj; last is NaN
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask.cpu(), expected))


if __name__ == "__main__":
    unittest.main()


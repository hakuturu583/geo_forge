import unittest

import numpy as np
import torch

from geo_forge.preprocess.sharp_util import _build_depth_supervision_mask


class TestSharpDepthSupervisionMask(unittest.TestCase):
    def test_excludes_sky_and_movable_objects(self) -> None:
        lidar_depth = torch.tensor(
            [
                [1.0, float("nan"), 3.0],
                [4.0, 100.0, float("nan")],
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
            lidar_depth=lidar_depth,
            sky_mask=sky,
            movable_object_mask=obj,
            max_depth=50.0,
            mask_upper_half=False,
        )

        expected = torch.tensor(
            [
                [True, False, False],  # third excluded by sky
                [True, False, False],  # middle excluded by obj/max_depth; last is NaN
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask.cpu(), expected))

    def test_mask_upper_half_excludes_top_rows(self) -> None:
        lidar_depth = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=torch.float32,
        )
        mask = _build_depth_supervision_mask(
            lidar_depth=lidar_depth,
            sky_mask=None,
            movable_object_mask=None,
            max_depth=None,
            mask_upper_half=True,
        )
        expected = torch.tensor(
            [
                [False, False],
                [False, False],
                [True, True],
                [True, True],
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(mask.cpu(), expected))


if __name__ == "__main__":
    unittest.main()

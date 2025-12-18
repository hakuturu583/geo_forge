import unittest

import numpy as np

from geo_forge.nuscenes import project_points_to_depth_image


class TestNuScenesDepthProjection(unittest.TestCase):
    def test_z_buffer_keeps_nearest_depth(self) -> None:
        width = 8
        height = 6
        fx = fy = 10.0
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

        # Two points that project to the same pixel (center), with different depths.
        points = np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        )

        depth, mask = project_points_to_depth_image(
            points_cam=points,
            intrinsics=K,
            width=width,
            height=height,
            min_depth=0.1,
            fill_value=0.0,
        )

        self.assertEqual(depth.shape, (height, width))
        self.assertEqual(mask.shape, (height, width))

        center_x = int(round(cx))
        center_y = int(round(cy))
        self.assertTrue(mask[center_y, center_x])
        self.assertAlmostEqual(float(depth[center_y, center_x]), 2.0, places=5)


if __name__ == "__main__":
    unittest.main()

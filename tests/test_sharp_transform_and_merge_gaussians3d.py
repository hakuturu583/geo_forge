import unittest

import gsplat
import torch
from sharp.utils.gaussians import Gaussians3D

from geo_forge.preprocess.sharp_util import merge_gaussians3d, transform_gaussians3d


def _c2w_from_rot_trans(rot_3x3: torch.Tensor, trans_3: torch.Tensor) -> torch.Tensor:
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = rot_3x3.to(dtype=torch.float32)
    c2w[:3, 3] = trans_3.to(dtype=torch.float32)
    return c2w


class TestTransformAndMergeGaussians3D(unittest.TestCase):
    def test_translation_only(self) -> None:
        gaussians_cam = Gaussians3D(
            mean_vectors=torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32),
            singular_values=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.tensor([[[0.5, 0.5, 0.5]]], dtype=torch.float32),
            opacities=torch.tensor([[0.8]], dtype=torch.float32),
        )
        c2w = _c2w_from_rot_trans(torch.eye(3), torch.tensor([10.0, -2.0, 3.0]))

        transformed = transform_gaussians3d(
            gaussians_camera=gaussians_cam, camera_c2w=c2w
        )
        expected_mean = torch.tensor([[[11.0, -2.0, 3.0]]], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(transformed.mean_vectors, expected_mean, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                transformed.quaternions,
                gaussians_cam.quaternions,
                atol=1e-6,
            )
        )

    def test_rotation_composition(self) -> None:
        gaussians_cam = Gaussians3D(
            mean_vectors=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
            singular_values=torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32),
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
            opacities=torch.tensor([[0.1]], dtype=torch.float32),
        )

        # 90 deg rotation around +Z.
        theta = torch.tensor(torch.pi / 2.0, dtype=torch.float32)
        rot = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0.0],
                [torch.sin(theta), torch.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        c2w = _c2w_from_rot_trans(rot, torch.zeros(3))

        transformed = transform_gaussians3d(
            gaussians_camera=gaussians_cam, camera_c2w=c2w
        )

        rot_out = gsplat.utils.normalized_quat_to_rotmat(transformed.quaternions[0, 0])
        self.assertTrue(torch.allclose(rot_out, rot, atol=1e-5))

    def test_merge_appends_gaussians(self) -> None:
        world_gaussians = Gaussians3D(
            mean_vectors=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
            singular_values=torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32),
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
            opacities=torch.tensor([[0.5]], dtype=torch.float32),
        )
        gaussians_cam = Gaussians3D(
            mean_vectors=torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32),
            singular_values=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
            quaternions=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
            colors=torch.tensor([[[0.9, 0.8, 0.7]]], dtype=torch.float32),
            opacities=torch.tensor([[0.9]], dtype=torch.float32),
        )
        c2w = torch.eye(4, dtype=torch.float32)

        transformed = transform_gaussians3d(
            gaussians_camera=gaussians_cam, camera_c2w=c2w
        )
        merged = merge_gaussians3d(base=world_gaussians, to_add=transformed)
        self.assertEqual(int(merged.mean_vectors.shape[1]), 2)
        self.assertTrue(
            torch.allclose(
                merged.mean_vectors[0, 0], world_gaussians.mean_vectors[0, 0]
            )
        )
        self.assertTrue(
            torch.allclose(merged.mean_vectors[0, 1], gaussians_cam.mean_vectors[0, 0])
        )


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import torch
from sharp.utils import color_space as color_space_utils
from sharp.utils.gaussians import Gaussians3D, convert_rgb_to_spherical_harmonics

from geo_forge.preprocess.sharp_preprocessor import gaussians3d_to_splatsim


def _quat_rotate_vector(
    quats_wxyz: torch.Tensor, vectors_xyz: torch.Tensor
) -> torch.Tensor:
    q_vec = quats_wxyz[..., 1:4]
    q_w = quats_wxyz[..., 0:1]
    uv = torch.cross(q_vec, vectors_xyz, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vectors_xyz + 2.0 * (q_w * uv + uuv)


class TestSharpSplatsimConversion(unittest.TestCase):
    def test_gaussians3d_to_splatsim_fields(self) -> None:
        means = torch.tensor(
            [[[1.0, 2.0, 3.0], [-1.5, 0.5, 10.0]]], dtype=torch.float32
        )
        singular_values = torch.tensor(
            [[[0.1, 0.2, 0.3], [1.0, 0.5, 2.0]]], dtype=torch.float32
        )
        quats = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.7071068, 0.7071068, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        quats = quats / quats.norm(dim=-1, keepdim=True)
        colors_linear = torch.tensor(
            [[[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]]], dtype=torch.float32
        )
        opacities = torch.tensor([[0.2, 0.8]], dtype=torch.float32)

        gaussians = Gaussians3D(
            mean_vectors=means,
            singular_values=singular_values,
            quaternions=quats,
            colors=colors_linear,
            opacities=opacities,
        )

        converted = gaussians3d_to_splatsim(gaussians)
        self.assertEqual(len(converted), 2)

        expected_scales_log = singular_values.log().flatten(0, 1)
        expected_opacity_logits = torch.logit(opacities).flatten(0, 1)
        expected_f_dc = convert_rgb_to_spherical_harmonics(
            color_space_utils.linearRGB2sRGB(colors_linear.flatten(0, 1))
        )

        z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        expected_normals = _quat_rotate_vector(
            quats.flatten(0, 1), z_axis.expand_as(means.flatten(0, 1))
        )

        for idx, gaussian in enumerate(converted):
            for a, b in zip(
                gaussian.position, means.flatten(0, 1)[idx].tolist(), strict=True
            ):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6))
            for a, b in zip(
                gaussian.scale, expected_scales_log[idx].tolist(), strict=True
            ):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6))
            self.assertTrue(
                math.isclose(
                    gaussian.opacity,
                    float(expected_opacity_logits[idx]),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            )
            for a, b in zip(
                gaussian.rot, quats.flatten(0, 1)[idx].tolist(), strict=True
            ):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6))
            for a, b in zip(gaussian.f_dc, expected_f_dc[idx].tolist(), strict=True):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6))
            for a, b in zip(
                gaussian.normal, expected_normals[idx].tolist(), strict=True
            ):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6))

            self.assertEqual(gaussian.f_rest, ())
            self.assertTrue(
                math.isclose(gaussian.reflection, 0.0, rel_tol=0.0, abs_tol=0.0)
            )


if __name__ == "__main__":
    unittest.main()

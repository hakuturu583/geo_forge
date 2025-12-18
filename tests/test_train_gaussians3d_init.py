import unittest

import torch
from sharp.utils.gaussians import Gaussians3D

from geo_forge.train.gs_train_config import GsTrainConfig
from geo_forge.train.train import _initialize_params


class _DummyDataset:
    pass


class TestTrainGaussians3DInit(unittest.TestCase):
    def test_initialize_params_from_gaussians3d(self) -> None:
        means = torch.tensor([[[1.0, 2.0, 3.0], [-1.5, 0.5, 10.0]]])
        singular_values = torch.tensor([[[0.1, 0.2, 0.3], [1.0, 0.5, 2.0]]])
        quats = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]])
        colors = torch.tensor([[[0.25, 0.5, 0.75], [0.1, 0.2, 0.3]]])
        opacities = torch.tensor([[0.2, 0.8]])

        gaussians = Gaussians3D(
            mean_vectors=means,
            singular_values=singular_values,
            quaternions=quats,
            colors=colors,
            opacities=opacities,
        )
        config = GsTrainConfig(num_gaussians=2)
        device = torch.device("cpu")

        params = _initialize_params(
            dataset=_DummyDataset(),
            config=config,
            device=device,
            init_gaussians=gaussians,
        )

        expected_means = means.flatten(0, 1).to(dtype=torch.float32)
        expected_scales = (
            singular_values.flatten(0, 1).to(dtype=torch.float32).clamp_min(1e-12).log()
        )
        expected_quats = quats.flatten(0, 1).to(dtype=torch.float32)
        expected_quats = expected_quats / expected_quats.norm(dim=-1, keepdim=True)
        expected_opacities = torch.logit(
            opacities.flatten(0, 1).to(dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
        )
        expected_colors = colors.flatten(0, 1).to(dtype=torch.float32)

        torch.testing.assert_close(params["means"].detach(), expected_means)
        torch.testing.assert_close(params["scales"].detach(), expected_scales)
        torch.testing.assert_close(params["quats"].detach(), expected_quats)
        torch.testing.assert_close(params["opacities"].detach(), expected_opacities)
        torch.testing.assert_close(params["colors"].detach(), expected_colors)


if __name__ == "__main__":
    unittest.main()

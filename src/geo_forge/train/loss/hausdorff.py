from __future__ import annotations

from geomloss import SamplesLoss
from geomloss.kernel_samples import kernel_routines
import torch

from geo_forge.train.loss.config import HausdorffLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class HausdorffLoss(LossBase):
    """
    Compute a Hausdorff distance between image-derived point sets.
    """

    name = "hausdorff"

    def __init__(self, config: HausdorffLossWeightConfig) -> None:
        super().__init__(config.schedule)
        self._config = config

    def compute(
        self,
        *,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample: dict[str, object] | None = None,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must share the same shape; "
                f"got {pred.shape} vs {target.shape}"
            )
        pred_points, pred_weights = self._sample_image_points(
            pred,
            max_points=self._config.max_points,
            threshold=self._config.threshold,
        )
        target_points, target_weights = self._sample_image_points(
            target,
            max_points=self._config.max_points,
            threshold=self._config.threshold,
        )
        if pred_points.numel() == 0 or target_points.numel() == 0:
            return pred.new_tensor(0.0)
        loss_fn = SamplesLoss(
            loss="hausdorff",
            p=2,
            blur=self._config.blur,
            kernel=kernel_routines["gaussian"],
        )
        loss = loss_fn(pred_weights, pred_points, target_weights, target_points)
        return self.apply_schedule(loss, step, total_steps)

    @staticmethod
    def _sample_image_points(
        image: torch.Tensor, *, max_points: int, threshold: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_points <= 0:
            raise ValueError("max_points must be positive.")
        if image.dim() == 3 and image.shape[0] == 3:
            luminance = (
                0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
            ).clamp_min(0.0)
        elif image.dim() == 2:
            luminance = image.clamp_min(0.0)
        else:
            raise ValueError(
                "Expected image shape (3, H, W) or (H, W); " f"got {tuple(image.shape)}"
            )

        height, width = luminance.shape
        weights = torch.where(luminance > threshold, luminance, luminance.new_zeros(()))
        flat_weights = weights.flatten()
        if flat_weights.sum().item() <= 0:
            return luminance.new_empty((0, 2)), luminance.new_empty((0,))

        nonzero = torch.nonzero(flat_weights > 0, as_tuple=False).squeeze(1)
        if nonzero.numel() == 0:
            return luminance.new_empty((0, 2)), luminance.new_empty((0,))
        if nonzero.numel() > max_points:
            sampled = torch.multinomial(
                flat_weights[nonzero], num_samples=max_points, replacement=False
            )
            indices = nonzero[sampled]
        else:
            indices = nonzero

        ys = indices // width
        xs = indices % width

        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        xs = xs.to(dtype=torch.float32) / float(denom_x)
        ys = ys.to(dtype=torch.float32) / float(denom_y)
        points = torch.stack([xs, ys], dim=1)
        weights = flat_weights[indices].to(dtype=torch.float32)
        weight_sum = weights.sum()
        if weight_sum.item() <= 0:
            return points.new_empty((0, 2)), points.new_empty((0,))
        weights = weights / weight_sum
        return points, weights

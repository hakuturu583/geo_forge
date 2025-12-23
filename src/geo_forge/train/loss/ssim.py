from __future__ import annotations

import torch
import torch.nn.functional as F

from geo_forge.train.loss.config import SSIMLossWeightConfig
from geo_forge.train.loss.loss_base import LossBase


class SSIMLoss(LossBase):
    """
    Structural Similarity (SSIM) loss for image reconstruction.
    """

    name = "ssim"

    def __init__(self, config: SSIMLossWeightConfig) -> None:
        super().__init__()
        self._window_size = config.window_size
        self._sigma = config.sigma
        self._data_range = config.data_range
        self._window_cache: dict[
            tuple[int, torch.device, torch.dtype], torch.Tensor
        ] = {}

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
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        if pred.dim() != 4:
            raise ValueError(
                "SSIM loss expects input with shape (C, H, W) or (N, C, H, W); "
                f"got {pred.shape}"
            )
        if self._window_size % 2 == 0 or self._window_size <= 0:
            raise ValueError("window_size must be a positive odd integer.")

        channels = pred.shape[1]
        window = self._get_window(
            channels=channels, device=pred.device, dtype=pred.dtype
        )
        padding = self._window_size // 2

        mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
        mu_target = F.conv2d(target, window, padding=padding, groups=channels)
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target

        sigma_pred = (
            F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
        ).clamp_min(0.0)
        sigma_target = (
            F.conv2d(target * target, window, padding=padding, groups=channels)
            - mu_target_sq
        ).clamp_min(0.0)
        sigma_pred_target = (
            F.conv2d(pred * target, window, padding=padding, groups=channels)
            - mu_pred_target
        )

        c1 = (0.01 * self._data_range) ** 2
        c2 = (0.03 * self._data_range) ** 2

        numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (
            sigma_pred + sigma_target + c2
        )
        ssim_map = numerator / denominator.clamp_min(1e-12)
        ssim_value = ssim_map.mean()
        return 1.0 - ssim_value

    def _get_window(
        self, *, channels: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (channels, device, dtype)
        cached = self._window_cache.get(key)
        if cached is not None:
            return cached

        coords = torch.arange(self._window_size, device=device, dtype=dtype)
        coords = coords - (self._window_size - 1) / 2.0
        gauss = torch.exp(-(coords ** 2) / (2.0 * self._sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = gauss[:, None] * gauss[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        window = kernel_2d.expand(channels, 1, self._window_size, self._window_size)
        window = window.contiguous()
        self._window_cache[key] = window
        return window

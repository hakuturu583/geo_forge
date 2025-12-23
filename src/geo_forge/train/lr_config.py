from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LRConfig:
    """
    Per-parameter learning rate multipliers for Gaussian splatting training.
    """

    base_lr: float = 5e-3
    means: float = 0.032
    scales: float = 1.0
    quats: float = 0.2
    opacities: float = 10.0
    colors: float = 0.5


__all__ = ["LRConfig"]

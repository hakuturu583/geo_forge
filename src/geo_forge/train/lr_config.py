from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LRValue:
    """
    Learning rate multiplier and optimizer epsilon for a parameter group.
    """

    value: float
    eps: float = 1e-15


@dataclass
class LRConfig:
    """
    Per-parameter learning rate multipliers for Gaussian splatting training.
    """

    base_lr: float = 5e-3
    means: LRValue = field(default_factory=lambda: LRValue(0.032))
    scales: LRValue = field(default_factory=lambda: LRValue(1.0))
    quats: LRValue = field(default_factory=lambda: LRValue(0.2))
    opacities: LRValue = field(default_factory=lambda: LRValue(10.0))
    colors: LRValue = field(default_factory=lambda: LRValue(0.5))

    @staticmethod
    def _parse_lr_value(raw: Any, default: LRValue) -> LRValue:
        if raw is None:
            return default
        if isinstance(raw, (int, float)):
            return LRValue(value=float(raw), eps=default.eps)
        if isinstance(raw, dict):
            if "value" not in raw:
                raise ValueError("lr_config entries must include a value field.")
            value = raw["value"]
            eps = raw.get("eps", default.eps)
            if not isinstance(value, (int, float)) or not isinstance(eps, (int, float)):
                raise ValueError("lr_config value and eps must be numeric.")
            return LRValue(value=float(value), eps=float(eps))
        raise ValueError("lr_config entries must be a number or a mapping.")

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "LRConfig":
        base_lr = raw.get("base_lr", cls.base_lr)
        if not isinstance(base_lr, (int, float)):
            raise ValueError("lr_config.base_lr must be numeric.")
        means_raw = raw.get("means")
        scales_raw = raw.get("scales")
        quats_raw = raw.get("quats", raw.get("quat"))
        opacities_raw = raw.get("opacities", raw.get("opaticies"))
        colors_raw = raw.get("colors")
        default = cls(base_lr=float(base_lr))
        return cls(
            base_lr=float(base_lr),
            means=cls._parse_lr_value(means_raw, default.means),
            scales=cls._parse_lr_value(scales_raw, default.scales),
            quats=cls._parse_lr_value(quats_raw, default.quats),
            opacities=cls._parse_lr_value(opacities_raw, default.opacities),
            colors=cls._parse_lr_value(colors_raw, default.colors),
        )


__all__ = ["LRConfig", "LRValue"]

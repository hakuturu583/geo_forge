from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import yaml

from geo_forge.train.gs_train_config import DefaultStrategyConfig, LossWeightConfig
from geo_forge.train.merge_prune_strategy import BackfacePruneConfig


@dataclass
class GsMergePruneConfig:
    """
    Lightweight config for the merge-prune SHARP Gaussians training loop.
    """

    steps: int = 500
    lr: float = 0.1
    scale_anisotropy_weight: float = 0.0
    scale_anisotropy_log_threshold: float = math.log(50.0)
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)
    strategy: DefaultStrategyConfig = field(default_factory=DefaultStrategyConfig)
    backface_prune: BackfacePruneConfig = field(default_factory=BackfacePruneConfig)
    device: str | None = None
    log_interval: int = 10

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GsMergePruneConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must define a mapping.")

        strategy_cfg: DefaultStrategyConfig
        backface_cfg: BackfacePruneConfig
        loss_weights_cfg: LossWeightConfig
        if "strategy" in raw:
            strategy_raw = raw.pop("strategy")
            if not isinstance(strategy_raw, dict):
                raise ValueError(
                    "strategy must be a mapping of DefaultStrategy values."
                )
            strategy_cfg = DefaultStrategyConfig(**strategy_raw)
        else:
            strategy_cfg = DefaultStrategyConfig()

        if "backface_prune" in raw:
            backface_raw = raw.pop("backface_prune")
            if isinstance(backface_raw, dict):
                backface_cfg = BackfacePruneConfig(**backface_raw)
            elif isinstance(backface_raw, bool):
                backface_cfg = BackfacePruneConfig(enabled=backface_raw)
            else:
                raise ValueError("backface_prune must be a bool or mapping.")
        else:
            backface_cfg = BackfacePruneConfig()

        if "loss_weights" in raw:
            loss_weights_raw = raw.pop("loss_weights")
            if not isinstance(loss_weights_raw, dict):
                raise ValueError(
                    "loss_weights must be a mapping of LossWeightConfig values."
                )
            loss_weights_cfg = LossWeightConfig(**loss_weights_raw)
        else:
            loss_weights_cfg = LossWeightConfig()

        return cls(
            strategy=strategy_cfg,
            backface_prune=backface_cfg,
            loss_weights=loss_weights_cfg,
            **raw,
        )


__all__ = ["GsMergePruneConfig"]

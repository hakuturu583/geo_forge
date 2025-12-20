from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from geo_forge.train.loss import (
    FrequencyDomainLossWeightConfig,
    LossWeightConfig,
    MaskLossWeightConfig,
)


@dataclass
class DefaultStrategyConfig:
    """
    Configuration for gsplat's DefaultStrategy growth/pruning behavior.
    """

    prune_opacity_threshold: float = 0.001
    prune_scale_threshold: float = 0.5
    grow_grad2d_threshold: float = 5e-5
    refine_start_iter: int = 250
    refine_stop_iter: int = 15000
    reset_every: int = 2000


@dataclass
class GsTrainConfig:
    """
    Configuration for the Gaussian splatting training demo.

    Populate from a YAML file to keep CLI usage minimal.
    """

    scenes: list[str] | None = None
    cameras: list[str] | None = None
    steps: int = 200
    num_gaussians: int = 8000
    lr: float = 5e-3
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)
    strategy: DefaultStrategyConfig = field(default_factory=DefaultStrategyConfig)
    device: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_interval: int = 10
    render_interval: int | None = None
    max_render_history: int | None = None
    max_eval_sets: int | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GsTrainConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must define a mapping.")
        strategy_cfg: DefaultStrategyConfig
        loss_weights_cfg: LossWeightConfig
        if "strategy" in raw:
            strategy_raw = raw.pop("strategy")
            if not isinstance(strategy_raw, dict):
                raise ValueError(
                    "strategy must be a mapping of DefaultStrategy values."
                )
            strategy_cfg = DefaultStrategyConfig(**strategy_raw)
        else:
            # Fallback: allow top-level strategy fields to keep older configs working.
            strategy_kwargs = {}
            for key in (
                "prune_opacity_threshold",
                "prune_scale_threshold",
                "grow_grad2d_threshold",
                "refine_start_iter",
                "refine_stop_iter",
                "reset_every",
            ):
                if key in raw:
                    strategy_kwargs[key] = raw.pop(key)
            strategy_cfg = (
                DefaultStrategyConfig(**strategy_kwargs)
                if strategy_kwargs
                else DefaultStrategyConfig()
            )
        if "loss_weights" in raw:
            loss_weights_raw = raw.pop("loss_weights")
            if not isinstance(loss_weights_raw, dict):
                raise ValueError(
                    "loss_weights must be a mapping with mask/frequency_domain."
                )
            mask_raw = loss_weights_raw.get("mask", {})
            frequency_raw = loss_weights_raw.get("frequency_domain", {})
            if not isinstance(mask_raw, dict) or not isinstance(frequency_raw, dict):
                raise ValueError(
                    "loss_weights.mask and loss_weights.frequency_domain must be mappings."
                )
            loss_weights_cfg = LossWeightConfig(
                mask=MaskLossWeightConfig(**mask_raw),
                frequency_domain=FrequencyDomainLossWeightConfig(**frequency_raw),
            )
        else:
            loss_weights_cfg = LossWeightConfig()
        return cls(strategy=strategy_cfg, loss_weights=loss_weights_cfg, **raw)


__all__ = ["DefaultStrategyConfig", "LossWeightConfig", "GsTrainConfig"]

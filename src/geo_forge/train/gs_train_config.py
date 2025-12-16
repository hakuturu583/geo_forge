from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DefaultStrategyConfig:
    """
    Configuration for gsplat's DefaultStrategy growth/pruning behavior.
    """

    prune_opacity_threshold: float = 0.001
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
        return cls(strategy=strategy_cfg, **raw)


__all__ = ["DefaultStrategyConfig", "GsTrainConfig"]

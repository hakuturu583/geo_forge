from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


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
    device: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_interval: int = 10
    render_interval: int | None = None
    max_render_history: int = 16
    max_eval_sets: int = 2

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GsTrainConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must define a mapping.")
        return cls(**raw)


__all__ = ["GsTrainConfig"]

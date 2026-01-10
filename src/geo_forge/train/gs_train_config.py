from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from geo_forge.train.loss import (
    AnisotropyLossWeightConfig,
    ChamferLossWeightConfig,
    EdgeAwareLossWeightConfig,
    FreeSpaceLossWeightConfig,
    FrequencyDomainLossWeightConfig,
    HausdorffLossWeightConfig,
    LossScheduleConfig,
    LossWeightConfig,
    MaskLossWeightConfig,
    OpacityLossWeightConfig,
    ScaleLossWeightConfig,
    SSIMLossWeightConfig,
)
from geo_forge.train.lr_config import LRConfig


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
    refine_every: int = 100


@dataclass
class LodConfig:
    """
    Distance-aware LOD settings for growth/pruning.
    """

    enabled: bool = False
    distance_threshold: float = 30.0
    update_interval: int = 1000
    query_chunk_size: int = 200000
    far_grad_scale: float = 0.2
    far_prune_opacity_multiplier: float = 2.0
    far_prune_scale_multiplier: float = 2.0


@dataclass
class DataLoaderConfig:
    """
    DataLoader settings for training.
    """

    num_workers: int = 0
    pin_memory: bool = False
    seed: int | None = None


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
    lr_config: LRConfig = field(default_factory=LRConfig)
    loss_weights: LossWeightConfig = field(default_factory=LossWeightConfig)
    strategy: DefaultStrategyConfig = field(default_factory=DefaultStrategyConfig)
    lod: LodConfig = field(default_factory=LodConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    device: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    log_interval: int = 10
    render_interval: int | None = None
    render_packed: bool = False
    max_render_history: int | None = None
    max_eval_sets: int | None = None

    @classmethod
    def from_raw(cls, raw: dict) -> "GsTrainConfig":
        if not isinstance(raw, dict):
            raise ValueError("GsTrainConfig raw config must be a mapping.")
        raw = dict(raw)
        raw.pop("hydra", None)
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
        if "lod" in raw:
            lod_raw = raw.pop("lod")
            if not isinstance(lod_raw, dict):
                raise ValueError("lod must be a mapping of LodConfig values.")
            lod_cfg = LodConfig(**lod_raw)
        else:
            lod_cfg = LodConfig()
        if "dataloader" in raw:
            dataloader_raw = raw.pop("dataloader")
            if not isinstance(dataloader_raw, dict):
                raise ValueError(
                    "dataloader must be a mapping of DataLoaderConfig values."
                )
            dataloader_cfg = DataLoaderConfig(**dataloader_raw)
        else:
            dataloader_cfg = DataLoaderConfig()
        if "loss_weights" in raw:
            loss_weights_raw = raw.pop("loss_weights")
            if not isinstance(loss_weights_raw, dict):
                raise ValueError(
                    "loss_weights must be a mapping with mask/frequency_domain."
                )
            mask_raw = loss_weights_raw.get("mask", {})
            frequency_raw = loss_weights_raw.get("frequency_domain", {})
            edge_raw = loss_weights_raw.get("edge_aware", {})
            opacity_raw = loss_weights_raw.get("opacity", {})
            scale_raw = loss_weights_raw.get("scale", {})
            anisotropy_raw = loss_weights_raw.get("anisotropy", {})
            free_space_raw = loss_weights_raw.get("free_space", {})
            ssim_raw = loss_weights_raw.get("ssim", {})
            hausdorff_raw = loss_weights_raw.get("hausdorff", {})
            chamfer_raw = loss_weights_raw.get("chamfer", {})
            free_space_schedule_raw = {}
            hausdorff_schedule_raw = {}
            chamfer_schedule_raw = {}
            if isinstance(free_space_raw, dict):
                free_space_schedule_raw = free_space_raw.get("schedule", {})
            if isinstance(hausdorff_raw, dict):
                hausdorff_schedule_raw = hausdorff_raw.get("schedule", {})
            if isinstance(chamfer_raw, dict):
                chamfer_schedule_raw = chamfer_raw.get("schedule", {})
            if (
                not isinstance(mask_raw, dict)
                or not isinstance(frequency_raw, dict)
                or not isinstance(edge_raw, dict)
                or not isinstance(opacity_raw, dict)
                or not isinstance(scale_raw, dict)
                or not isinstance(anisotropy_raw, dict)
                or not isinstance(free_space_raw, dict)
                or not isinstance(ssim_raw, dict)
                or not isinstance(hausdorff_raw, dict)
                or not isinstance(free_space_schedule_raw, dict)
                or not isinstance(hausdorff_schedule_raw, dict)
                or not isinstance(chamfer_raw, dict)
                or not isinstance(chamfer_schedule_raw, dict)
            ):
                raise ValueError(
                    "loss_weights.mask, loss_weights.frequency_domain, and "
                    "loss_weights.edge_aware, loss_weights.opacity, "
                    "loss_weights.scale, loss_weights.anisotropy, "
                    "loss_weights.free_space, "
                    "loss_weights.ssim, "
                    "loss_weights.hausdorff, and loss_weights.chamfer "
                    "must be mappings."
                )
            hausdorff_kwargs = dict(hausdorff_raw)
            hausdorff_kwargs.pop("schedule", None)
            chamfer_kwargs = dict(chamfer_raw)
            chamfer_kwargs.pop("schedule", None)
            free_space_kwargs = dict(free_space_raw)
            free_space_kwargs.pop("schedule", None)
            loss_weights_cfg = LossWeightConfig(
                mask=MaskLossWeightConfig(**mask_raw),
                frequency_domain=FrequencyDomainLossWeightConfig(**frequency_raw),
                edge_aware=EdgeAwareLossWeightConfig(**edge_raw),
                opacity=OpacityLossWeightConfig(**opacity_raw),
                scale=ScaleLossWeightConfig(**scale_raw),
                anisotropy=AnisotropyLossWeightConfig(**anisotropy_raw),
                free_space=FreeSpaceLossWeightConfig(
                    **free_space_kwargs,
                    schedule=LossScheduleConfig(**free_space_schedule_raw),
                ),
                ssim=SSIMLossWeightConfig(**ssim_raw),
                hausdorff=HausdorffLossWeightConfig(
                    **hausdorff_kwargs,
                    schedule=LossScheduleConfig(**hausdorff_schedule_raw),
                ),
                chamfer=ChamferLossWeightConfig(
                    **chamfer_kwargs,
                    schedule=LossScheduleConfig(**chamfer_schedule_raw),
                ),
            )
        else:
            loss_weights_cfg = LossWeightConfig()
        lr_cfg: LRConfig
        if "lr_config" in raw:
            lr_raw = raw.pop("lr_config")
            if not isinstance(lr_raw, dict):
                raise ValueError("lr_config must be a mapping of LRConfig values.")
            lr_cfg = LRConfig.from_raw(lr_raw)
            raw.pop("lr", None)
        else:
            base_lr = raw.pop("lr", None)
            if base_lr is None:
                lr_cfg = LRConfig()
            else:
                lr_cfg = LRConfig(base_lr=base_lr)
        return cls(
            strategy=strategy_cfg,
            lod=lod_cfg,
            dataloader=dataloader_cfg,
            loss_weights=loss_weights_cfg,
            lr_config=lr_cfg,
            lr=lr_cfg.base_lr,
            **raw,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "GsTrainConfig":
        with Path(path).open("r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {path} must define a mapping.")
        return cls.from_raw(raw)


__all__ = [
    "DefaultStrategyConfig",
    "DataLoaderConfig",
    "LodConfig",
    "LossWeightConfig",
    "GsTrainConfig",
    "LRConfig",
]

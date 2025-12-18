from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

from geo_forge.dataset import GeoForgeDataset
from geo_forge.nuscenes import lidar_depth_from_synchronized_sample
from geo_forge.preprocess.preprocess import resolve_dataset_root
from geo_forge.preprocess.sharp_util import (
    OptimizeScaleConfig,
    gaussians3d_to_splatsim,
    optimize_scale,
)
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    save_ply,
    unproject_gaussians,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

load_dotenv()


@dataclass
class SharpPreprocessorConfig:
    """
    Configuration for running SHARP Gaussian prediction over ROSE outputs.

    The fields mirror the upstream SHARP CLI flags but are loaded from YAML for
    easier reproducibility.
    """

    scenes: list[str] | None = None
    cameras: list[str] | None = None
    output_root: Path | str | None = None
    checkpoint_path: Path | str | None = None
    device: str = "default"
    verbose: bool = False
    max_frames: int | None = None
    optimize_scale: OptimizeScaleConfig = field(default_factory=OptimizeScaleConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SharpPreprocessorConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {config_path} must define a mapping.")

        # Normalize optional list fields that may be provided as scalars.
        def _ensure_list(value: object) -> list[str] | None:
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip()
                return [value] if value else None
            if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
                normalized: list[str] = []
                for v in value:
                    text = str(v).strip()
                    if text:
                        normalized.append(text)
                return normalized or None
            raise ValueError("Expected list or string for list fields.")

        scenes = _ensure_list(raw.get("scenes"))
        cameras = _ensure_list(raw.get("cameras"))

        remaining = {k: v for k, v in raw.items() if k not in {"scenes", "cameras"}}

        optimize_scale_raw = remaining.pop("optimize_scale", None)
        optimize_scale: OptimizeScaleConfig | None = None
        if optimize_scale_raw is None:
            legacy = {
                k: v for k, v in remaining.items() if k.startswith("optimize_scale_")
            }
            if legacy:
                for key in list(legacy.keys()):
                    remaining.pop(key, None)
                mapped: dict[str, object] = {}
                mapping = {
                    "optimize_scale_steps": "steps",
                    "optimize_scale_lr": "lr",
                    "optimize_scale_init": "init_scale",
                    "optimize_scale_min_scale": "min_scale",
                    "optimize_scale_max_scale": "max_scale",
                    "optimize_scale_min_depth": "min_depth",
                    "optimize_scale_max_depth": "max_depth",
                    "optimize_scale_tile_size": "tile_size",
                    "optimize_scale_near_plane": "near_plane",
                    "optimize_scale_far_plane": "far_plane",
                }
                for legacy_key, new_key in mapping.items():
                    if legacy_key in legacy:
                        mapped[new_key] = legacy[legacy_key]
                enabled = legacy.get("optimize_scale")
                if enabled is not None:
                    mapped["enabled"] = bool(enabled)
                optimize_scale = OptimizeScaleConfig(**mapped)
        elif isinstance(optimize_scale_raw, dict):
            optimize_scale = OptimizeScaleConfig(**optimize_scale_raw)
        else:
            raise ValueError("optimize_scale must be a mapping if provided.")

        return cls(
            scenes=scenes,
            cameras=cameras,
            optimize_scale=optimize_scale or OptimizeScaleConfig(),
            **remaining,
        )


def _select_device(device_pref: str) -> torch.device:
    if device_pref in (None, "", "default"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_pref)


def _load_predictor(
    checkpoint_path: Path | None, device: torch.device
) -> RGBGaussianPredictor:
    if checkpoint_path is None:
        LOGGER.info(
            "No checkpoint provided. Downloading default model from %s",
            DEFAULT_MODEL_URL,
        )
        state_dict = torch.hub.load_state_dict_from_url(
            DEFAULT_MODEL_URL, progress=True
        )
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def _prepare_image_tensor(image: torch.Tensor) -> np.ndarray:
    """
    Convert a CHW float tensor in [0, 1] to an HWC numpy array in [0, 255].
    """
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Expected image tensor shape (3, H, W); got {tuple(image.shape)}"
        )
    image_np = (
        image.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    return image_np


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    intrinsics: torch.Tensor,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image using NuScenes intrinsics (adapted from SHARP CLI)."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = (
        torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    )
    _, height, width = image_pt.shape

    intrinsics_3x3 = intrinsics.to(device=device, dtype=torch.float32)
    f_x = float(intrinsics_3x3[0, 0])
    f_y = float(intrinsics_3x3[1, 1])
    c_x = float(intrinsics_3x3[0, 2])
    c_y = float(intrinsics_3x3[1, 2])
    f_px = float((f_x + f_y) / 2.0)
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics_full = torch.eye(4, device=device, dtype=torch.float32)
    intrinsics_full[0, 0] = f_x
    intrinsics_full[1, 1] = f_y
    intrinsics_full[0, 2] = c_x
    intrinsics_full[1, 2] = c_y

    intrinsics_resized = intrinsics_full.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


def run_sharp_preprocess(config: SharpPreprocessorConfig) -> None:
    logging_utils.configure(logging.DEBUG if config.verbose else logging.INFO)

    device = _select_device(config.device)
    if config.optimize_scale.enabled and device.type != "cuda":
        raise RuntimeError(
            "optimize_scale requires CUDA, but the selected device is "
            f"{device.type!r}. Set device='cuda' or disable optimize_scale."
        )

    checkpoint_path = Path(config.checkpoint_path) if config.checkpoint_path else None
    predictor = _load_predictor(checkpoint_path, device=device)

    output_root_env = os.getenv("GEOFORGE_DATASET_ROOT")
    output_root = resolve_dataset_root(
        override=output_root_env or config.output_root,
        env_var="GEOFORGE_DATASET_ROOT",
    )
    scene_filter = config.scenes or None
    camera_filter = config.cameras or None
    if scene_filter is None:
        LOGGER.info("No scenes specified; defaulting to all scenes.")
    dataset = GeoForgeDataset(
        scene_filter=scene_filter,
        camera_filter=camera_filter,
        only_sample_frames=True,
    )

    LOGGER.info("Processing %d frames from GeoForgeDataset.", len(dataset))
    processed = 0
    for sample in dataset:
        if config.max_frames is not None and processed >= config.max_frames:
            break

        scene = sample["scene"]
        camera = sample["camera"]
        timestamp = sample["timestamp"]
        width = int(sample["width"])
        height = int(sample["height"])

        image_np = _prepare_image_tensor(sample["image"])
        intrinsics = sample["intrinsics"]
        f_px = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
        gaussians = predict_image(
            predictor,
            image_np,
            intrinsics=intrinsics,
            device=device,
        )

        if config.optimize_scale.enabled:
            sample_token = sample.get("nusc_sample_token")
            if not isinstance(sample_token, str) or not sample_token:
                raise RuntimeError(
                    "GeoForgeDataset did not provide 'nusc_sample_token' for scale optimization."
                )
            sample_record = dataset.nusc.get("sample", sample_token)
            camera_name = str(camera).upper()
            cam_token = sample_record["data"][camera_name]
            cam_data = dataset.nusc.get("sample_data", cam_token)
            lidar_token = sample_record["data"]["LIDAR_TOP"]
            lidar_data = dataset.nusc.get("sample_data", lidar_token)

            sample_info = {
                "sample_token": sample_token,
                "scene_name": scene,
                "timestamp": int(lidar_data["timestamp"]),
                "lidar": {
                    "token": lidar_token,
                    "filename": lidar_data["filename"],
                    "timestamp": int(lidar_data["timestamp"]),
                    "calibrated_sensor_token": lidar_data["calibrated_sensor_token"],
                    "ego_pose_token": lidar_data["ego_pose_token"],
                },
                "cameras": {
                    camera_name: {
                        "token": cam_token,
                        "filename": cam_data["filename"],
                        "timestamp": int(cam_data["timestamp"]),
                        "calibrated_sensor_token": cam_data["calibrated_sensor_token"],
                        "ego_pose_token": cam_data["ego_pose_token"],
                    }
                },
                "is_key_frame": True,
            }
            lidar_depth, _, _ = lidar_depth_from_synchronized_sample(
                dataset.nusc,
                sample_info,
                camera_name=camera_name,
                dataroot=dataset.dataroot,
                image_size=(width, height),
                min_depth=float(config.optimize_scale.min_depth),
                max_depth=(
                    float(config.optimize_scale.max_depth)
                    if config.optimize_scale.max_depth is not None
                    else None
                ),
                fill_value=float("nan"),
            )
            try:
                c2w_identity = torch.eye(4, dtype=torch.float32)
                gaussians, scale, losses = optimize_scale(
                    gaussians=gaussians,
                    lidar_depth=lidar_depth,
                    intrinsics=intrinsics,
                    c2w=c2w_identity,
                    sky_mask=sample.get("sky_mask"),
                    movable_object_mask=sample.get("object_mask"),
                    max_depth=config.optimize_scale.max_depth,
                    steps=int(config.optimize_scale.steps),
                    lr=float(config.optimize_scale.lr),
                    init_scale=float(config.optimize_scale.init_scale),
                    min_scale=float(config.optimize_scale.min_scale),
                    max_scale=float(config.optimize_scale.max_scale),
                    tile_size=int(config.optimize_scale.tile_size),
                    near_plane=float(config.optimize_scale.near_plane),
                    far_plane=float(config.optimize_scale.far_plane),
                    device=device,
                    verbose=config.verbose,
                )
            except ValueError as exc:
                LOGGER.warning(
                    "Scale optimization skipped for %s/%s/%s: %s",
                    scene,
                    camera,
                    timestamp,
                    exc,
                )
            else:
                LOGGER.info(
                    "Optimized scale for %s/%s/%s: scale=%.6f final_loss=%.6f valid_steps=%d",
                    scene,
                    camera,
                    timestamp,
                    float(scale),
                    float(losses[-1]) if losses else float("nan"),
                    len(losses),
                )

        output_dir = output_root / scene / camera / "sharp"
        output_dir.mkdir(parents=True, exist_ok=True)
        ply_path = output_dir / f"{timestamp}_sharp.ply"
        LOGGER.info("Saving 3DGS to %s", ply_path)
        save_ply(gaussians, f_px, (height, width), ply_path)

        processed += 1

    LOGGER.info("Finished processing %d frames.", processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict Gaussians with SHARP using ROSE object-removed frames "
            "loaded from GeoForgeDataset."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML file containing SharpPreprocessorConfig values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SharpPreprocessorConfig.from_yaml(args.config)
    run_sharp_preprocess(config)


if __name__ == "__main__":
    main()

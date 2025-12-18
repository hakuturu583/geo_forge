from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv

from geo_forge.dataset import GeoForgeDataset
from geo_forge.preprocess.preprocess import resolve_dataset_root
from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import logging as logging_utils
from sharp.utils import color_space as color_space_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    convert_rgb_to_spherical_harmonics,
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
        return cls(scenes=scenes, cameras=cameras, **remaining)


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


def gaussians3d_to_splatsim(gaussians: Gaussians3D) -> list["Gaussian"]:
    """
    Convert a SHARP ``Gaussians3D`` to a splatsim ``list[Gaussian]``.

    Notes:
        - ``Gaussian.scale`` is stored in log-space (matching 3DGS PLY exporters).
        - ``Gaussian.opacity`` is stored as a logit (inverse-sigmoid).
        - ``Gaussian.f_dc`` follows SHARP's PLY exporter: linearRGB -> sRGB -> SH-DC.
        - Fields unused by our training demo (``normal``, ``f_rest``, ``reflection``)
          are still populated with reasonable defaults.
    """
    from splatsim import Gaussian

    xyz = gaussians.mean_vectors.flatten(0, 1).detach().to(dtype=torch.float32)
    scales_log = (
        gaussians.singular_values.flatten(0, 1)
        .detach()
        .to(dtype=torch.float32)
        .clamp_min(1e-12)
        .log()
    )
    quaternions = gaussians.quaternions.flatten(0, 1).detach().to(dtype=torch.float32)
    opacities = gaussians.opacities.flatten(0, 1).detach().to(dtype=torch.float32)
    opacity_logits = torch.logit(opacities.clamp(1e-6, 1.0 - 1e-6)).squeeze(-1)

    # SHARP predicts linearRGB, while most downstream tools expect sRGB-like values.
    colors_sh_dc = convert_rgb_to_spherical_harmonics(
        color_space_utils.linearRGB2sRGB(
            gaussians.colors.flatten(0, 1).detach().to(dtype=torch.float32)
        )
    )

    def _quat_rotate_vector(
        quats_wxyz: torch.Tensor, vectors_xyz: torch.Tensor
    ) -> torch.Tensor:
        q_vec = quats_wxyz[..., 1:4]
        q_w = quats_wxyz[..., 0:1]
        uv = torch.cross(q_vec, vectors_xyz, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vectors_xyz + 2.0 * (q_w * uv + uuv)

    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=xyz.device)
    normals = _quat_rotate_vector(quaternions, z_axis.expand_as(xyz)).cpu().numpy()

    xyz_np = xyz.cpu().numpy()
    scales_np = scales_log.cpu().numpy()
    quats_np = quaternions.cpu().numpy()
    op_np = opacity_logits.cpu().numpy()
    colors_np = colors_sh_dc.cpu().numpy()

    splat_gaussians: list[Gaussian] = []
    for idx in range(xyz_np.shape[0]):
        splat_gaussians.append(
            Gaussian(
                position=tuple(float(v) for v in xyz_np[idx]),
                normal=tuple(float(v) for v in normals[idx]),
                f_dc=tuple(float(v) for v in colors_np[idx]),
                f_rest=(),
                opacity=float(op_np[idx]),
                scale=tuple(float(v) for v in scales_np[idx]),
                rot=tuple(float(v) for v in quats_np[idx]),
                reflection=0.0,
            )
        )
    return splat_gaussians


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

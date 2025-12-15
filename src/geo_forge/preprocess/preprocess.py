"""Shared preprocessing utilities for SAM3 demo scripts."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Sequence

import imageio.v3 as iio
import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

from geo_forge.dataclass import ObjectMask

load_dotenv()


def resolve_dataset_root(
    override: Path | str | None = None, env_var: str = "GEOFORGE_DATASET_ROOT"
) -> Path:
    """
    Resolve the datasets output root, preferring an explicit override, then an
    environment variable, and finally the default ``preprocess/datasets`` path.
    """
    env_root = os.getenv(env_var)
    base_dir = Path(__file__).resolve().parent

    if override is not None:
        root = Path(override)
    elif env_root:
        root = Path(env_root)
    else:
        root = base_dir / "datasets"

    root = root.expanduser()
    if not root.is_absolute():
        root = base_dir / root
    return root


def save_mask_artifacts(
    output_dir: Path,
    file_stem: str,
    mask: ObjectMask,
    metadata: Dict[str, Any],
) -> None:
    """Persist mask tensor (npz) and metadata (json)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_payload = {"masks": mask.masks.cpu().numpy()}
    if mask.scores is not None:
        mask_payload["scores"] = mask.scores.cpu().numpy()
    if mask.labels is not None:
        mask_payload["labels"] = mask.labels.cpu().numpy()
    if mask.boxes is not None:
        mask_payload["boxes"] = mask.boxes.cpu().numpy()

    np.savez_compressed(output_dir / f"{file_stem}_mask.npz", **mask_payload)

    metadata_path = output_dir / f"{file_stem}_meta.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def combine_layer_masks(
    mask_objects: list[ObjectMask],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    Collapse a collection of ObjectMask predictions into a single layer mask.
    """
    width, height = image_size
    expected_shape = (height, width)

    layer_masks: list[torch.Tensor] = []
    for mask_obj in mask_objects:
        if mask_obj.masks.numel() == 0:
            continue

        mask_tensor = mask_obj.masks.detach().to("cpu")
        if mask_tensor.dim() == 3:
            layer_mask = mask_tensor.any(dim=0)
        elif mask_tensor.dim() == 2:
            layer_mask = mask_tensor.bool()
        else:
            raise ValueError(f"Unsupported mask dimensionality: {mask_tensor.dim()}")

        if layer_mask.shape != expected_shape:
            raise ValueError(
                "Mask and image spatial dimensions do not match: "
                f"mask={layer_mask.shape[::-1]}, image={(width, height)}"
            )

        layer_masks.append(layer_mask)

    if not layer_masks:
        return torch.zeros(expected_shape, dtype=torch.bool)

    return torch.stack(layer_masks).any(dim=0)


def save_layer_mask(
    layer_dir: Path, file_stem: str, layer_name: str, mask_tensor: torch.Tensor
) -> Path:
    """Persist a boolean mask tensor for a logical layer."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    mask_path = layer_dir / f"{file_stem}_{layer_name}.pt"
    torch.save(mask_tensor.cpu(), mask_path)
    return mask_path


def apply_mask_to_image(image: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
    """Apply a boolean mask to an image and return the masked copy."""
    masked_image = image.convert("RGB")
    mask_np = mask_tensor.bool().cpu().numpy()
    width, height = masked_image.size
    if mask_np.shape != (height, width):
        raise ValueError(
            "Mask and image spatial dimensions do not match: "
            f"mask={mask_np.shape[::-1]}, image={masked_image.size}"
        )

    img_arr = np.array(masked_image, copy=True)
    img_arr[mask_np] = 0
    return Image.fromarray(img_arr)


def export_video_from_frames(
    frames: Sequence[Image.Image] | Sequence[np.ndarray],
    output_path: Path | str,
    fps: int = 8,
) -> Path:
    """
    Write a sequence of RGB frames to an mp4 video using imageio.
    """
    if not frames:
        raise ValueError("No frames provided to export_video_from_frames")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_arrays: list[np.ndarray] = []
    expected_size: tuple[int, int] | None = None
    for frame in frames:
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[2] not in (3, 4):
            raise ValueError("Frames must be RGB or RGBA images")
        if frame_array.shape[2] == 4:
            frame_array = frame_array[:, :, :3]

        height, width, _ = frame_array.shape
        if expected_size is None:
            expected_size = (height, width)
        elif expected_size != (height, width):
            raise ValueError(
                "All frames must share dimensions for video export: "
                f"expected={expected_size[::-1]}, got={(width, height)}"
            )

        frame_arrays.append(frame_array)

    write_kwargs: dict[str, Any] = {"fps": fps}
    suffix = output_path.suffix.lower()
    if suffix in {".mp4", ".mov", ".mkv"}:
        write_kwargs["codec"] = "h264"
    if suffix == ".gif":
        write_kwargs["loop"] = 0

    iio.imwrite(output_path, frame_arrays, **write_kwargs)
    return output_path


__all__ = [
    "apply_mask_to_image",
    "combine_layer_masks",
    "export_video_from_frames",
    "save_layer_mask",
    "save_mask_artifacts",
]

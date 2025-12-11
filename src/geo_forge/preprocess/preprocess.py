"""
Sample preprocessing pipeline for NuScenes frames using SAM3 masks.

Loads NuScenes samples, generates attribute masks, serializes mask tensors, and
writes visualizations under src/geo_forge/preprocess/datasets by default.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Sequence
from collections import defaultdict

import imageio.v3 as iio
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image

from geo_forge.preprocess.sam3_preprocessor import SAM3Preprocessor
from geo_forge.preprocess.sam3_video_preprocessor import SAM3VideoPreprocessor
from geo_forge.nuscenes import iterate_synchronized_samples, load_synchronized_data
from geo_forge.dataclass import ObjectMask, overray_mask


def _save_mask_artifacts(
    output_dir: Path,
    file_stem: str,
    mask: ObjectMask,
    metadata: Dict[str, Any],
) -> None:
    """Persist mask tensor (npz) and metadata (json)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize mask tensor and optional fields
    mask_payload = {"masks": mask.masks.cpu().numpy()}
    if mask.scores is not None:
        mask_payload["scores"] = mask.scores.cpu().numpy()
    if mask.labels is not None:
        mask_payload["labels"] = mask.labels.cpu().numpy()
    if mask.boxes is not None:
        mask_payload["boxes"] = mask.boxes.cpu().numpy()

    np.savez_compressed(output_dir / f"{file_stem}_mask.npz", **mask_payload)

    # Save metadata alongside artifacts
    metadata_path = output_dir / f"{file_stem}_meta.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def _combine_layer_masks(
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


def _save_layer_mask(
    layer_dir: Path, file_stem: str, layer_name: str, mask_tensor: torch.Tensor
) -> Path:
    """Persist a boolean mask tensor for a logical layer."""
    layer_dir.mkdir(parents=True, exist_ok=True)
    mask_path = layer_dir / f"{file_stem}_{layer_name}.pt"
    torch.save(mask_tensor.cpu(), mask_path)
    return mask_path


def _apply_mask_to_image(image: Image.Image, mask_tensor: torch.Tensor) -> Image.Image:
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


def run_preprocess(
    max_samples: int | None = None,
    output_root: Path | None = None,
    scene_names: list[str] | None = None,
) -> None:
    """
    Run a lightweight preprocessing demo over NuScenes frames.

    Generates attribute masks, serializes mask data, and writes masked images.
    Defaults to the first NuScenes mini scene to keep runtime short.
    """
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    preprocessor = SAM3Preprocessor()

    # Process only the first mini scene by default to keep the demo lightweight.
    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [nusc.scene[0]["name"]]

    print(f"Processing scenes: {', '.join(scene_names)}")

    if output_root is None:
        output_root = Path(__file__).resolve().parent / "datasets"

    raw_frames_by_camera: dict[tuple[str, str], list[Image.Image]] = defaultdict(list)
    sky_masked_frames_by_camera: dict[tuple[str, str], list[Image.Image]] = defaultdict(
        list
    )
    video_frames_by_camera: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for sample_idx, sample_info in enumerate(
        iterate_synchronized_samples(nusc, scene_names=scene_names)
    ):
        if max_samples is not None and sample_idx >= max_samples:
            break

        _, images, boxes = load_synchronized_data(nusc, sample_info)
        scene_dir = output_root / sample_info["scene_name"]

        for cam_name, cam_data in images.items():
            image = cam_data["image"]
            width, height = image.size
            cam_dir = scene_dir / cam_name.lower()
            cam_dir.mkdir(parents=True, exist_ok=True)
            file_stem = f"{sample_info['timestamp']}_{cam_name.lower()}"
            # raw_path = (
            #     cam_dir / f"{sample_info['timestamp']}_{cam_name.lower()}_raw.jpg"
            # )
            # image.save(raw_path)
            raw_frames_by_camera[(sample_info["scene_name"], cam_name)].append(
                image.copy()
            )
            combined_masks: list[ObjectMask] = []
            attr_masks = preprocessor.generate_attribute_mask(image, "sky")
            combined_masks.extend(attr_masks)
            sky_layer = _combine_layer_masks(attr_masks, (width, height))
            sky_path = _save_layer_mask(cam_dir, file_stem, "sky", sky_layer)
            print(f"Saved sky layer mask for {cam_name} to {sky_path}")
            sky_masked_frames_by_camera[(sample_info["scene_name"], cam_name)].append(
                _apply_mask_to_image(image, sky_layer)
            )
            video_frames_by_camera[(sample_info["scene_name"], cam_name)].append(
                {
                    "image": image,
                    "timestamp": sample_info["timestamp"],
                    "scene_name": sample_info["scene_name"],
                }
            )

    for (scene_name, cam_name), frames in raw_frames_by_camera.items():
        raw_gif_path = (
            output_root / scene_name / cam_name.lower() / f"{cam_name.lower()}_raw.gif"
        )
        export_video_from_frames(frames, raw_gif_path)
        print(f"Saved raw frames GIF for {cam_name} to {raw_gif_path}")

    for (scene_name, cam_name), frames in sky_masked_frames_by_camera.items():
        sky_gif_path = (
            output_root
            / scene_name
            / cam_name.lower()
            / f"{cam_name.lower()}_sky_mask.gif"
        )
        export_video_from_frames(frames, sky_gif_path)
        print(f"Saved sky-masked GIF for {cam_name} to {sky_gif_path}")

    # Propagate prompts across the collected frames for each camera.
    if video_frames_by_camera:
        video_preprocessor = SAM3VideoPreprocessor()
        video_prompts = ["vehicle", "pedestrian", "bicycle", "animal"]
        for (scene_name, cam_name), frames in video_frames_by_camera.items():
            print(f"Generating video masks for {cam_name} across {len(frames)} frames")
            object_masks = video_preprocessor.generate_masks_from_video(
                [frame["image"] for frame in frames], video_prompts
            )
            masked_frames: list[Image.Image] = []
            for i, frame in enumerate(frames):
                masks = object_masks[i]
                scene_dir = output_root / frame["scene_name"]
                cam_dir = scene_dir / cam_name.lower()
                cam_dir.mkdir(parents=True, exist_ok=True)
                file_stem = f"{frame['timestamp']}_{cam_name.lower()}"
                masked_image = overray_mask(frame["image"], masks)
                masked_frames.append(masked_image)

                width, height = frame["image"].size
                movable_layer = _combine_layer_masks(masks, (width, height))
                movable_path = _save_layer_mask(
                    cam_dir, file_stem, "movable_objects", movable_layer
                )
                print(f"Saved movable_objects layer for {cam_name} to {movable_path}")
            video_path = (
                output_root
                / scene_name
                / cam_name.lower()
                / f"{cam_name.lower()}_movable_layer_mask.gif"
            )
            export_video_from_frames(masked_frames, video_path)
            print(f"Saved video masks for {cam_name} to {video_path}")


if __name__ == "__main__":
    run_preprocess()

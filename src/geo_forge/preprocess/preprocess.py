"""
Sample preprocessing pipeline for NuScenes frames using SAM3 masks.

Loads NuScenes samples, generates attribute masks, serializes mask tensors, and
writes visualizations under src/geo_forge/preprocess/datasets by default.
"""

from pathlib import Path
import os
import json
from typing import Dict, Any

import numpy as np
from nuscenes.nuscenes import NuScenes

from geo_forge.preprocess.sam3_preprocessor import SAM3Preprocessor
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


def run_preprocess(
    attribute_prompt: str = "sky",
    max_samples: int = 1,
    output_root: Path | None = None,
) -> None:
    """
    Run a lightweight preprocessing demo over NuScenes frames.

    Generates attribute masks, serializes mask data, and writes masked images.
    """
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    preprocessor = SAM3Preprocessor()

    if output_root is None:
        output_root = Path(__file__).resolve().parent / "datasets"

    for sample_idx, sample_info in enumerate(iterate_synchronized_samples(nusc)):
        if sample_idx >= max_samples:
            break

        _, images, boxes = load_synchronized_data(nusc, sample_info)
        scene_dir = output_root / sample_info["scene_name"]

        for cam_name, cam_data in images.items():
            image = cam_data["image"]
            cam_dir = scene_dir / cam_name.lower()
            cam_dir.mkdir(parents=True, exist_ok=True)
            raw_path = (
                cam_dir / f"{sample_info['timestamp']}_{cam_name.lower()}_raw.jpg"
            )
            image.save(raw_path)
            combined_masks: list[ObjectMask] = []
            attr_masks = preprocessor.generate_attribute_mask(image, attribute_prompt)
            for idx, mask_obj in enumerate(attr_masks):
                combined_masks.append(mask_obj)
                print(f"Saved attribute masks for {cam_name} to {scene_dir}")

            # Generate masks from 3D boxes if present
            box_masks = preprocessor.generate_masks_from_boxes(
                nusc, cam_data, image, boxes
            )
            print(f"Generated {len(box_masks)} box masks for {cam_name}")
            for idx, mask_obj in enumerate(box_masks):
                combined_masks.append(mask_obj)
                masked_image = overray_mask(image, combined_masks)
                viz_path = (
                    cam_dir
                    / f"{sample_info['timestamp']}_{cam_name.lower()}_combined_viz.jpg"
                )
                masked_image.save(viz_path)
                print(f"Saved combined mask visualization for {cam_name} to {viz_path}")


if __name__ == "__main__":
    run_preprocess()

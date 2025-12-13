import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from geo_forge.nuscenes import iterate_synchronized_samples, load_synchronized_data
from geo_forge.preprocess.preprocess import (
    apply_mask_to_image,
    combine_layer_masks,
    export_video_from_frames,
    save_layer_mask,
)


class SAM3Preprocessor:
    def __init__(self, model_name: str = "facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def _run_inference(self, inputs: Any) -> List[ObjectMask]:
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        return ObjectMask.from_result_list(results)

    def generate_attribute_mask(
        self, image: Image.Image, attribute_prompt: str | List[str]
    ) -> List[ObjectMask]:
        """
        Generate segmentation masks for attribute prompts using SAM3.

        Accepts a single prompt or a list of prompts; when a list is provided,
        masks from each prompt are concatenated.
        """
        if isinstance(attribute_prompt, (list, tuple)):
            combined_masks: List[ObjectMask] = []
            for prompt in attribute_prompt:
                combined_masks.extend(self.generate_attribute_mask(image, prompt))
            return combined_masks

        inputs = self.processor(
            images=image, text=attribute_prompt, return_tensors="pt"
        ).to(self.device)
        return self._run_inference(inputs)

    def generate_masks_from_boxes(
        self,
        nusc: NuScenes,
        camera_data: Dict[str, Any],
        image: Image.Image,
        boxes: Sequence[NuscenesObjectBoundingBox],
    ) -> List[ObjectMask]:
        """
        Generate segmentation masks for the given NuScenes bounding boxes.

        Projects 3D boxes into the camera plane and uses them as box prompts.
        """
        width, height = image.size
        sam_condition_boxes: List[List[float]] = []
        for box in boxes:
            bbox_2d = box.to_2d_bbox(
                nusc=nusc,
                calibrated_sensor_token=camera_data["calibrated_sensor_token"],
                ego_pose_token=camera_data["ego_pose_token"],
                image_size=(width, height),
                ignore_category=[
                    "movable_object.barrier",
                    "movable_object.trafficcone",
                    "movable_object.pushable_pullable",
                    "movable_object.debris",
                    "static_object.bicycle_rack",
                ],
            )
            if bbox_2d is None:
                continue

            x_min, y_min, x_max, y_max = bbox_2d
            x0, x1 = int(np.floor(x_min)), int(np.ceil(x_max))
            y0, y1 = int(np.floor(y_min)), int(np.ceil(y_max))
            sam_condition_boxes.append([x0, y0, x1, y1])

        if not sam_condition_boxes:
            return []

        inputs = self.processor(
            images=image,
            input_boxes=[sam_condition_boxes],
            input_boxes_labels=[[1] * len(sam_condition_boxes)],
            return_tensors="pt",
        ).to(self.device)
        return self._run_inference(inputs)


def run_sam3_attribute_preprocess(
    max_samples: int | None = None,
    output_root: Path | None = None,
    scene_names: list[str] | None = None,
) -> None:
    """
    Run a lightweight attribute-masking demo over NuScenes frames.

    Generates sky masks, serializes mask data, and writes masked/visualization GIFs.
    """
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    preprocessor = SAM3Preprocessor()

    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [scene["name"] for scene in nusc.scene]

    if output_root is None:
        output_root = Path(__file__).resolve().parent / "datasets"
    else:
        output_root = Path(output_root)

    raw_frames_by_camera: dict[tuple[str, str], list[Image.Image]] = defaultdict(list)
    sky_masked_frames_by_camera: dict[tuple[str, str], list[Image.Image]] = defaultdict(
        list
    )

    print(f"Processing scenes: {', '.join(scene_names)}")
    for sample_idx, sample_info in enumerate(
        iterate_synchronized_samples(nusc, scene_names=scene_names)
    ):
        if max_samples is not None and sample_idx >= max_samples:
            break

        _, images, _ = load_synchronized_data(nusc, sample_info)
        for cam_name, cam_data in images.items():
            image = cam_data["image"]
            width, height = image.size
            file_stem = f"{sample_info['timestamp']}_{cam_name.lower()}"

            raw_frames_by_camera[(sample_info["scene_name"], cam_name)].append(
                image.copy()
            )
            attr_masks = preprocessor.generate_attribute_mask(image, "sky")
            sky_layer = combine_layer_masks(attr_masks, (width, height))
            sky_mask_dir = (
                output_root / sample_info["scene_name"] / cam_name.lower() / "mask"
            )
            sky_path = save_layer_mask(sky_mask_dir, file_stem, "sky", sky_layer)
            print(f"Saved sky layer mask for {cam_name} to {sky_path}")

            sky_masked_frames_by_camera[(sample_info["scene_name"], cam_name)].append(
                apply_mask_to_image(image, sky_layer)
            )

    for (scene_name, cam_name), frames in raw_frames_by_camera.items():
        cam_visualization_dir = (
            output_root / scene_name / cam_name.lower() / "visualization"
        )
        raw_gif_path = cam_visualization_dir / f"{cam_name.lower()}_raw.gif"
        export_video_from_frames(frames, raw_gif_path)
        print(f"Saved raw frames GIF for {cam_name} to {raw_gif_path}")

    for (scene_name, cam_name), frames in sky_masked_frames_by_camera.items():
        cam_visualization_dir = (
            output_root / scene_name / cam_name.lower() / "visualization"
        )
        sky_gif_path = cam_visualization_dir / f"{cam_name.lower()}_sky_mask.gif"
        export_video_from_frames(frames, sky_gif_path)
        print(f"Saved sky-masked GIF for {cam_name} to {sky_gif_path}")


if __name__ == "__main__":
    run_sam3_attribute_preprocess()

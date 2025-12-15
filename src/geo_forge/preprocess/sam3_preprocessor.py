import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
import yaml
from transformers import Sam3Model, Sam3Processor

from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from geo_forge.nuscenes import (
    iterate_all_sweep_camera_frames,
    iterate_synchronized_samples,
    load_synchronized_data,
)
from geo_forge.preprocess.preprocess import (
    apply_mask_to_image,
    combine_layer_masks,
    export_video_from_frames,
    resolve_dataset_root,
    save_layer_mask,
)


@dataclass
class Sam3PromptLayerConfig:
    """Configuration describing a logical mask layer and its prompts."""

    layer_name: str
    prompts: list[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sam3PromptLayerConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Prompt config not found: {config_path}")

        with config_path.open() as f:
            raw_config = yaml.safe_load(f) or {}

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"Prompt config must be a mapping (got {type(raw_config).__name__})"
            )

        layer_name = raw_config.get("layer_name")
        raw_prompts = raw_config.get("prompts") or raw_config.get("prompt")

        if not layer_name:
            raise ValueError("Prompt config missing required key: layer_name")
        if raw_prompts is None:
            raise ValueError("Prompt config missing required key: prompts")

        if isinstance(raw_prompts, str):
            prompts = [raw_prompts]
        elif isinstance(raw_prompts, Sequence) and not isinstance(
            raw_prompts, (bytes, str)
        ):
            prompts = [str(prompt) for prompt in raw_prompts if prompt]
        else:
            raise ValueError(
                "prompts must be a string or a sequence of strings in prompt config"
            )

        if not prompts:
            raise ValueError("Prompt config must provide at least one prompt string")

        return cls(layer_name=str(layer_name), prompts=prompts)


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
        self, image: Image.Image, attribute_prompt: str | Sequence[str]
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
    camera_names: list[str] | None = None,
    only_sample_frames: bool = True,
    prompt_config_path: str | Path | None = None,
) -> None:
    """
    Run a lightweight attribute-masking demo over NuScenes frames.

    Prompts and the resulting layer name are read from a YAML config. When
    ``only_sample_frames`` is False, sweep frames between keyframes are also
    processed to produce masks for every available camera frame. Mask tensors
    and visualization GIFs are written under ``output_root``.
    """
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    preprocessor = SAM3Preprocessor()

    repo_root = Path(__file__).resolve().parents[3]
    default_config = repo_root / "configs" / "sam3" / "sky.yaml"
    prompt_config = Sam3PromptLayerConfig.from_yaml(
        prompt_config_path if prompt_config_path is not None else default_config
    )

    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [scene["name"] for scene in nusc.scene]

    output_root = resolve_dataset_root(output_root)
    camera_whitelist = [cam.upper() for cam in camera_names] if camera_names else None
    target_cameras = camera_whitelist or [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]

    print(
        f"Processing scenes: {', '.join(scene_names)}"
        + (f" | cameras: {', '.join(sorted(cam.lower() for cam in target_cameras))}")
    )

    for cam_name in target_cameras:
        cam_key = cam_name.lower()
        raw_frames_by_scene: dict[str, list[Image.Image]] = defaultdict(list)
        masked_frames_by_scene: dict[str, list[Image.Image]] = defaultdict(list)
        frame_iterator = (
            iterate_synchronized_samples(
                nusc, scene_names=scene_names, cameras=[cam_name]
            )
            if only_sample_frames
            else iterate_all_sweep_camera_frames(
                nusc, scene_names=scene_names, cameras=[cam_name]
            )
        )

        print(f"Processing camera: {cam_name}")

        def flush_camera(scene_name: str) -> None:
            if not scene_name:
                return

            raw_frames = raw_frames_by_scene.get(scene_name, [])
            masked_frames = masked_frames_by_scene.get(scene_name, [])
            if raw_frames:
                cam_visualization_dir = (
                    output_root / scene_name / cam_key / "visualization"
                )
                raw_gif_path = cam_visualization_dir / f"{cam_key}_raw.gif"
                export_video_from_frames(raw_frames, raw_gif_path)
                print(
                    f"Saved raw frames GIF for {cam_key} in {scene_name} to {raw_gif_path}"
                )
            if masked_frames:
                cam_visualization_dir = (
                    output_root / scene_name / cam_key / "visualization"
                )
                layer_gif_path = (
                    cam_visualization_dir
                    / f"{cam_key}_{prompt_config.layer_name}_mask.gif"
                )
                export_video_from_frames(masked_frames, layer_gif_path)
                print(
                    f"Saved {prompt_config.layer_name}-masked GIF for {cam_key} in {scene_name} to {layer_gif_path}"
                )

        current_scene: str | None = None
        try:
            for sample_idx, sample_info in enumerate(frame_iterator):
                if max_samples is not None and sample_idx >= max_samples:
                    break

                scene_name = sample_info["scene_name"]
                if current_scene is None:
                    current_scene = scene_name
                elif scene_name != current_scene:
                    flush_camera(current_scene)
                    raw_frames_by_scene.clear()
                    masked_frames_by_scene.clear()
                    current_scene = scene_name

                _, images, _ = load_synchronized_data(nusc, sample_info)
                cam_data = images.get(cam_name)
                if cam_data is None:
                    continue

                image = cam_data["image"]
                width, height = image.size
                file_stem = f"{sample_info['timestamp']}_{cam_key}"

                raw_frames_by_scene[scene_name].append(image.copy())
                attr_masks = preprocessor.generate_attribute_mask(
                    image, prompt_config.prompts
                )
                sky_layer = combine_layer_masks(attr_masks, (width, height))
                layer_mask_dir = output_root / scene_name / cam_key / "mask"
                mask_path = save_layer_mask(
                    layer_mask_dir, file_stem, prompt_config.layer_name, sky_layer
                )
                print(
                    f"Saved {prompt_config.layer_name} layer mask for {cam_key} in {scene_name} to {mask_path}"
                )

                masked_frames_by_scene[scene_name].append(
                    apply_mask_to_image(image, sky_layer)
                )
        finally:
            if current_scene is not None:
                flush_camera(current_scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAM3 attribute preprocessing over NuScenes frames."
    )
    parser.add_argument(
        "--scene",
        "-s",
        action="append",
        dest="scenes",
        help="Scene name to process (repeatable). Defaults to all scenes.",
    )
    parser.add_argument(
        "--camera",
        "-c",
        action="append",
        dest="cameras",
        help="Camera name to process (repeatable). Defaults to all cameras.",
    )
    parser.add_argument(
        "--only-sample-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limit processing to keyframe samples (default). Disable to include sweep frames.",
    )
    parser.add_argument(
        "--config",
        "-f",
        dest="config_path",
        help="Path to a YAML prompt config (defaults to config/sam3/sky.yaml).",
    )
    args = parser.parse_args()
    run_sam3_attribute_preprocess(
        scene_names=args.scenes,
        camera_names=args.cameras,
        only_sample_frames=args.only_sample_frames,
        prompt_config_path=args.config_path,
    )

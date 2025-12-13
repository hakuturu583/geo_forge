import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from accelerate import Accelerator
from nuscenes.nuscenes import NuScenes
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig

from geo_forge.dataclass import ObjectMask, overray_mask
from geo_forge.nuscenes import iterate_synchronized_samples, load_synchronized_data
from geo_forge.preprocess.preprocess import (
    combine_layer_masks,
    export_video_from_frames,
    save_layer_mask,
)


class SAM3VideoPreprocessor:
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        dtype: torch.dtype | None = None,
        config: Sam3VideoConfig | None = None,
    ):
        self.device = Accelerator().device
        if dtype is None:
            # Use bfloat16 on supported CUDA devices, otherwise fall back to float32.
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self.dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and bf16_supported
                else torch.float32
            )
        else:
            self.dtype = dtype

        self.model = Sam3VideoModel.from_pretrained(model_name).to(
            self.device, dtype=self.dtype
        )
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.accelerator = Accelerator()

    def generate_masks_from_video(
        self,
        video_frames: List[torch.Tensor],
        prompts: List[str],
    ) -> List[List[ObjectMask]]:
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(inference_session, prompts)
        object_masks: List[List[ObjectMask]] = []
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=50
        ):
            processed_outputs = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            object_masks_per_frame: List[ObjectMask] = []
            object_masks_per_frame.append(
                ObjectMask(
                    masks=processed_outputs["masks"], boxes=processed_outputs["boxes"]
                )
            )

            object_masks.append(object_masks_per_frame)
        return object_masks


def run_sam3_video_preprocess(
    max_samples: int | None = None,
    output_root: Path | None = None,
    scene_names: list[str] | None = None,
    camera_names: list[str] | None = None,
    config: Sam3VideoConfig | None = None,
) -> None:
    """
    Propagate prompts across NuScenes video frames and export movable masks.
    """
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [scene["name"] for scene in nusc.scene]

    if output_root is None:
        output_root = Path(__file__).resolve().parent / "datasets"
    else:
        output_root = Path(output_root)
    camera_filter = {cam.lower() for cam in camera_names} if camera_names else None

    video_frames_by_camera: dict[
        tuple[str, str], list[dict[str, Image.Image]]
    ] = defaultdict(list)

    print(
        f"Collecting frames for scenes: {', '.join(scene_names)}"
        + (f" | cameras: {', '.join(sorted(camera_filter))}" if camera_filter else "")
    )
    for sample_idx, sample_info in enumerate(
        iterate_synchronized_samples(nusc, scene_names=scene_names)
    ):
        if max_samples is not None and sample_idx >= max_samples:
            break

        _, images, _ = load_synchronized_data(nusc, sample_info)
        for cam_name, cam_data in images.items():
            cam_key = cam_name.lower()
            if camera_filter and cam_key not in camera_filter:
                continue

            video_frames_by_camera[(sample_info["scene_name"], cam_key)].append(
                {
                    "image": cam_data["image"],
                    "timestamp": sample_info["timestamp"],
                    "scene_name": sample_info["scene_name"],
                }
            )

    if not video_frames_by_camera:
        raise RuntimeError("No video frames were collected for processing")

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
            movable_layer = combine_layer_masks(masks, (width, height))
            movable_mask_dir = (
                output_root / frame["scene_name"] / cam_name.lower() / "mask"
            )
            movable_path = save_layer_mask(
                movable_mask_dir, file_stem, "movable_objects", movable_layer
            )
            print(f"Saved movable_objects layer for {cam_name} to {movable_path}")

        video_path = (
            output_root
            / scene_name
            / cam_name.lower()
            / "visualization"
            / f"{cam_name.lower()}_movable_layer_mask.gif"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        export_video_from_frames(masked_frames, video_path)
        print(f"Saved video masks for {cam_name} to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SAM3 video preprocessing over NuScenes frames."
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
    args = parser.parse_args()
    config = Sam3VideoConfig(
        score_threshold_detection=0.2,
        new_det_thresh=0.4,
        det_nms_thresh=0.3,
    )
    run_sam3_video_preprocess(
        scene_names=args.scenes, camera_names=args.cameras, config=config
    )

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
from geo_forge.nuscenes import (
    iterate_all_sweep_camera_frames,
    iterate_synchronized_samples,
    load_synchronized_data,
)
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

    def init_streaming_session(self, prompts: List[str]):
        """
        Initialize a streaming inference session and attach prompts.
        """
        session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(session, prompts)
        return session

    def stream_video_frame(self, session, frame_image: Image.Image) -> List[ObjectMask]:
        """
        Run streaming inference on a single frame and return ObjectMask list.
        """
        inputs = self.processor(
            images=frame_image, device=self.device, return_tensors="pt"
        )
        model_outputs = self.model(
            inference_session=session,
            frame=inputs.pixel_values[0],
            reverse=False,
        )
        processed_outputs = self.processor.postprocess_outputs(
            session,
            model_outputs,
            original_sizes=inputs.original_sizes,
        )

        masks = processed_outputs.get("masks")
        if masks is None or masks.numel() == 0:
            return []

        boxes = processed_outputs.get("boxes")
        scores = processed_outputs.get("scores")
        labels = (
            processed_outputs.get("object_ids")
            if processed_outputs.get("object_ids") is not None
            else processed_outputs.get("labels")
        )

        object_masks: List[ObjectMask] = []
        if masks.dim() == 3:
            num_instances = masks.shape[0]
            for idx in range(num_instances):
                object_masks.append(
                    ObjectMask(
                        masks=masks[idx],
                        scores=scores[idx] if scores is not None else None,
                        labels=labels[idx] if labels is not None else None,
                        boxes=boxes[idx] if boxes is not None else None,
                    )
                )
        else:
            object_masks.append(
                ObjectMask(
                    masks=masks,
                    scores=scores,
                    labels=labels,
                    boxes=boxes,
                )
            )

        return object_masks

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
    only_sample_frames: bool = True,
) -> None:
    """
    Propagate prompts across NuScenes video frames and export movable masks.

    When ``only_sample_frames`` is False, sweep frames between keyframes are
    also collected and masked, enabling per-camera video coverage.
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
        f"Collecting frames for scenes: {', '.join(scene_names)}"
        + (f" | cameras: {', '.join(sorted(camera_filter))}" if camera_filter else "")
    )
    video_prompts = ["Vehicle", "Pedestrian", "Bicycle", "Cyclist", "Animal"]

    if only_sample_frames:
        processed_any_camera = False
        for cam_name in target_cameras:
            video_preprocessor = SAM3VideoPreprocessor()
            frame_iterator = iterate_synchronized_samples(
                nusc,
                scene_names=scene_names,
                cameras=[cam_name],
            )
            frames_by_scene: dict[str, list[dict[str, Image.Image]]] = defaultdict(list)

            for sample_idx, sample_info in enumerate(frame_iterator):
                if max_samples is not None and sample_idx >= max_samples:
                    break

                _, images, _ = load_synchronized_data(nusc, sample_info)
                cam_data = images.get(cam_name)
                if cam_data is None:
                    continue

                frames_by_scene[sample_info["scene_name"]].append(
                    {
                        "image": cam_data["image"],
                        "timestamp": sample_info["timestamp"],
                        "scene_name": sample_info["scene_name"],
                    }
                )

            if not frames_by_scene:
                continue

            processed_any_camera = True
            cam_key = cam_name.lower()
            for scene_name, frames in frames_by_scene.items():
                print(
                    f"Generating video masks for {cam_key} in {scene_name} across {len(frames)} frames"
                )
                object_masks = video_preprocessor.generate_masks_from_video(
                    [frame["image"] for frame in frames], video_prompts
                )
                masked_frames: list[Image.Image] = []
                for i, frame in enumerate(frames):
                    masks = object_masks[i]
                    scene_dir = output_root / scene_name
                    cam_dir = scene_dir / cam_key
                    cam_dir.mkdir(parents=True, exist_ok=True)
                    file_stem = f"{frame['timestamp']}_{cam_key}"
                    masked_image = overray_mask(frame["image"], masks)
                    masked_frames.append(masked_image)

                    width, height = frame["image"].size
                    movable_layer = combine_layer_masks(masks, (width, height))
                    movable_mask_dir = output_root / scene_name / cam_key / "mask"
                    movable_path = save_layer_mask(
                        movable_mask_dir, file_stem, "movable_objects", movable_layer
                    )
                    print(
                        f"Saved movable_objects layer for {cam_key} to {movable_path}"
                    )

                video_path = (
                    output_root
                    / scene_name
                    / cam_key
                    / "visualization"
                    / f"{cam_key}_movable_layer_mask.gif"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                export_video_from_frames(masked_frames, video_path)
                print(f"Saved video masks for {cam_key} to {video_path}")

        if not processed_any_camera:
            raise RuntimeError("No video frames were collected for processing")
    else:
        for cam_name in target_cameras:
            video_preprocessor = SAM3VideoPreprocessor()
            cam_key = cam_name.lower()
            frame_iterator = iterate_all_sweep_camera_frames(
                nusc, scene_names=scene_names, cameras=[cam_name]
            )
            masked_frames: list[Image.Image] = []
            current_scene: str | None = None
            session = None

            for sample_idx, sample_info in enumerate(frame_iterator):
                if max_samples is not None and sample_idx >= max_samples:
                    break

                scene_name = sample_info["scene_name"]
                if current_scene is None:
                    current_scene = scene_name
                elif scene_name != current_scene:
                    if masked_frames:
                        video_path = (
                            output_root
                            / current_scene
                            / cam_key
                            / "visualization"
                            / f"{cam_key}_movable_layer_mask.gif"
                        )
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        export_video_from_frames(masked_frames, video_path)
                        print(f"Saved video masks for {cam_key} to {video_path}")
                    masked_frames = []
                    session = None
                    current_scene = scene_name

                _, images, _ = load_synchronized_data(nusc, sample_info)
                cam_data = images.get(cam_name)
                if cam_data is None:
                    continue

                if session is None:
                    session = video_preprocessor.init_streaming_session(video_prompts)

                masks = video_preprocessor.stream_video_frame(
                    session, cam_data["image"]
                )
                masked_image = overray_mask(cam_data["image"], masks)
                masked_frames.append(masked_image)

                width, height = cam_data["image"].size
                file_stem = f"{sample_info['timestamp']}_{cam_key}"
                movable_layer = combine_layer_masks(masks, (width, height))
                movable_mask_dir = (
                    output_root / sample_info["scene_name"] / cam_key / "mask"
                )
                movable_path = save_layer_mask(
                    movable_mask_dir, file_stem, "movable_objects", movable_layer
                )
                print(f"Saved movable_objects layer for {cam_key} to {movable_path}")

            if current_scene is not None and masked_frames:
                video_path = (
                    output_root
                    / current_scene
                    / cam_key
                    / "visualization"
                    / f"{cam_key}_movable_layer_mask.gif"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                export_video_from_frames(masked_frames, video_path)
                print(f"Saved video masks for {cam_key} to {video_path}")


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
    parser.add_argument(
        "--only-sample-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Limit processing to keyframe samples (default). Disable to include sweep frames.",
    )
    args = parser.parse_args()
    config = Sam3VideoConfig(
        score_threshold_detection=0.2,
        new_det_thresh=0.4,
        det_nms_thresh=0.3,
        fill_hole_area=5,
    )
    run_sam3_video_preprocess(
        scene_names=args.scenes,
        camera_names=args.cameras,
        config=config,
        only_sample_frames=args.only_sample_frames,
    )

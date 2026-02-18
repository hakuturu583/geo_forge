import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from nuscenes.nuscenes import NuScenes
from PIL import Image
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor

from geo_forge.dataclass import NuscenesObjectBoundingBox, ObjectMask
from geo_forge.preprocess.preprocess import (
    apply_mask_to_image,
    export_video_from_frames,
    resolve_dataset_root,
    save_layer_mask,
)

DEFAULT_CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

load_dotenv()


@dataclass
class SAM3MovableObjectPreprocessorConfig:
    model_name: str = "facebook/sam3"
    speed_threshold_kmh: float = 1.0
    visibility_token_max: int = 2
    max_missing_frames: int = 2
    iou_threshold: float = 0.01
    fps: int = 8


class SAM3MovableObjectPreprocessor:
    """Track movable NuScenes instances with SAM3 video segmentation."""

    def __init__(
        self,
        config: SAM3MovableObjectPreprocessorConfig | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.config = config or SAM3MovableObjectPreprocessorConfig()
        self.device = Accelerator().device

        if dtype is None:
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self.dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and bf16_supported
                else torch.float32
            )
        else:
            self.dtype = dtype

        self.model = Sam3VideoModel.from_pretrained(self.config.model_name).to(
            self.device, dtype=self.dtype
        )
        self.processor = Sam3VideoProcessor.from_pretrained(self.config.model_name)

    def init_streaming_session(self, prompt: str):
        session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(session, [prompt])
        return session

    def stream_video_frame(
        self,
        session: Any,
        frame_image: Image.Image,
        reverse: bool = False,
    ) -> list[ObjectMask]:
        inputs = self.processor(
            images=frame_image, device=self.device, return_tensors="pt"
        )
        outputs = self.model(
            inference_session=session,
            frame=inputs.pixel_values[0],
            reverse=reverse,
        )
        processed = self.processor.postprocess_outputs(
            session,
            outputs,
            original_sizes=inputs.original_sizes,
        )

        masks = processed.get("masks")
        if masks is None or masks.numel() == 0:
            return []

        boxes = processed.get("boxes")
        scores = processed.get("scores")
        labels = (
            processed.get("object_ids")
            if processed.get("object_ids") is not None
            else processed.get("labels")
        )

        object_masks: list[ObjectMask] = []
        if masks.dim() == 3:
            for idx in range(masks.shape[0]):
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
                ObjectMask(masks=masks, scores=scores, labels=labels, boxes=boxes)
            )
        return object_masks

    @staticmethod
    def _safe_speed_kmh(nusc: NuScenes, ann_token: str) -> float:
        velocity = nusc.box_velocity(ann_token)
        if velocity is None:
            return 0.0
        vx, vy = float(velocity[0]), float(velocity[1])
        if not math.isfinite(vx) or not math.isfinite(vy):
            return 0.0
        return math.sqrt(vx * vx + vy * vy) * 3.6

    @staticmethod
    def _flatten_masks(mask_objects: list[ObjectMask]) -> list[torch.Tensor]:
        masks: list[torch.Tensor] = []
        for mask_obj in mask_objects:
            if mask_obj.masks.numel() == 0:
                continue
            mask_t = mask_obj.masks.detach().to("cpu")
            if mask_t.dim() == 2:
                masks.append(mask_t.bool())
            elif mask_t.dim() == 3:
                for idx in range(mask_t.shape[0]):
                    masks.append(mask_t[idx].bool())
        return masks

    @staticmethod
    def _bbox_to_mask(
        bbox: tuple[float, float, float, float], image_size: tuple[int, int]
    ) -> torch.Tensor:
        width, height = image_size
        x_min, y_min, x_max, y_max = bbox
        x0 = max(0, min(width, int(np.floor(x_min))))
        y0 = max(0, min(height, int(np.floor(y_min))))
        x1 = max(0, min(width, int(np.ceil(x_max))))
        y1 = max(0, min(height, int(np.ceil(y_max))))

        mask = torch.zeros((height, width), dtype=torch.bool)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
        return mask

    @staticmethod
    def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
        inter = torch.logical_and(mask_a, mask_b).sum().item()
        union = torch.logical_or(mask_a, mask_b).sum().item()
        if union == 0:
            return 0.0
        return float(inter / union)

    def _select_best_mask(
        self,
        candidates: list[ObjectMask],
        image_size: tuple[int, int],
        bbox: tuple[float, float, float, float] | None = None,
        reference_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        flat = self._flatten_masks(candidates)
        if not flat:
            return None

        target_mask: torch.Tensor | None = None
        if bbox is not None:
            target_mask = self._bbox_to_mask(bbox, image_size)
        elif reference_mask is not None:
            target_mask = reference_mask.bool().to("cpu")

        if target_mask is None:
            return flat[0]

        best_mask: torch.Tensor | None = None
        best_iou = -1.0
        for cand in flat:
            if cand.shape != target_mask.shape:
                continue
            iou = self._mask_iou(cand, target_mask)
            if iou > best_iou:
                best_iou = iou
                best_mask = cand

        if best_mask is None or best_iou < self.config.iou_threshold:
            return None
        return best_mask

    @staticmethod
    def _build_scene_camera_frames(
        nusc: NuScenes,
        scene_name: str,
        camera_name: str,
    ) -> list[dict[str, Any]]:
        scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
        if scene is None:
            raise ValueError(f"Scene not found: {scene_name}")

        first_sample = nusc.get("sample", scene["first_sample_token"])
        cam_token = first_sample["data"][camera_name]

        frames: list[dict[str, Any]] = []
        while cam_token:
            cam_sd = nusc.get("sample_data", cam_token)
            frames.append(
                {
                    "token": cam_token,
                    "timestamp": int(cam_sd["timestamp"]),
                    "filename": str(cam_sd["filename"]),
                    "is_key_frame": bool(cam_sd["is_key_frame"]),
                    "sample_token": cam_sd.get("sample_token"),
                    "calibrated_sensor_token": cam_sd["calibrated_sensor_token"],
                    "ego_pose_token": cam_sd["ego_pose_token"],
                }
            )
            cam_token = cam_sd.get("next", "")
        return frames

    @staticmethod
    def _load_frame_image(nusc: NuScenes, frame: dict[str, Any]) -> Image.Image:
        frame_path = Path(nusc.dataroot) / frame["filename"]
        with Image.open(frame_path) as img:
            return img.convert("RGB")

    @staticmethod
    def _instance_sample_annotation_map(
        nusc: NuScenes,
        instance_token: str,
        scene_token: str,
    ) -> dict[str, dict[str, Any]]:
        ann_tokens = nusc.field2token(
            "sample_annotation", "instance_token", instance_token
        )
        mapping: dict[str, dict[str, Any]] = {}
        for ann_token in ann_tokens:
            ann = nusc.get("sample_annotation", ann_token)
            sample = nusc.get("sample", ann["sample_token"])
            if sample["scene_token"] != scene_token:
                continue
            mapping[ann["sample_token"]] = ann
        return mapping

    def _track_instance_for_camera(
        self,
        nusc: NuScenes,
        scene_token: str,
        instance_token: str,
        frame_infos: list[dict[str, Any]],
    ) -> dict[int, torch.Tensor]:
        sample_to_ann = self._instance_sample_annotation_map(
            nusc, instance_token=instance_token, scene_token=scene_token
        )
        if not sample_to_ann:
            return {}

        frame_idx_to_bbox: dict[int, tuple[float, float, float, float]] = {}
        frame_idx_to_prompt: dict[int, str] = {}

        for idx, frame in enumerate(frame_infos):
            if not frame["is_key_frame"]:
                continue
            sample_token = frame.get("sample_token")
            ann = sample_to_ann.get(sample_token)
            if ann is None:
                continue

            image = self._load_frame_image(nusc, frame)
            box = NuscenesObjectBoundingBox.from_sample_annotation(ann)
            bbox = box.to_2d_bbox(
                nusc=nusc,
                calibrated_sensor_token=frame["calibrated_sensor_token"],
                ego_pose_token=frame["ego_pose_token"],
                image_size=image.size,
            )
            if bbox is None:
                continue

            frame_idx_to_bbox[idx] = bbox
            frame_idx_to_prompt[idx] = str(ann.get("category_name", "object")).replace(
                ".", " "
            )

        if not frame_idx_to_bbox:
            return {}

        first_idx = min(frame_idx_to_bbox.keys())
        last_idx = max(frame_idx_to_bbox.keys())
        prompt = frame_idx_to_prompt[first_idx]

        tracked: dict[int, torch.Tensor] = {}

        # Forward tracking: keyframe interval + post-disappearance sweeps.
        session = self.init_streaming_session(prompt)
        prev_mask: torch.Tensor | None = None
        missing_after_last = 0
        for idx in range(first_idx, len(frame_infos)):
            frame = frame_infos[idx]
            image = self._load_frame_image(nusc, frame)
            candidates = self.stream_video_frame(session, image, reverse=False)
            selected = self._select_best_mask(
                candidates=candidates,
                image_size=image.size,
                bbox=frame_idx_to_bbox.get(idx),
                reference_mask=prev_mask,
            )

            if selected is not None:
                tracked[idx] = selected
                prev_mask = selected
                missing_after_last = 0
            elif idx > last_idx:
                missing_after_last += 1
                if missing_after_last > self.config.max_missing_frames:
                    break

        # Backward tracking: pre-appearance sweeps until disappearance.
        session_back = self.init_streaming_session(prompt)
        anchor_frame = frame_infos[first_idx]
        anchor_image = self._load_frame_image(nusc, anchor_frame)
        anchor_candidates = self.stream_video_frame(
            session_back, anchor_image, reverse=False
        )
        anchor_mask = self._select_best_mask(
            candidates=anchor_candidates,
            image_size=anchor_image.size,
            bbox=frame_idx_to_bbox.get(first_idx),
            reference_mask=None,
        )

        prev_back_mask = anchor_mask
        if anchor_mask is not None:
            tracked[first_idx] = anchor_mask

        missing_before_first = 0
        for idx in range(first_idx - 1, -1, -1):
            frame = frame_infos[idx]
            image = self._load_frame_image(nusc, frame)
            candidates = self.stream_video_frame(session_back, image, reverse=True)
            selected = self._select_best_mask(
                candidates=candidates,
                image_size=image.size,
                bbox=frame_idx_to_bbox.get(idx),
                reference_mask=prev_back_mask,
            )

            if selected is not None:
                tracked[idx] = selected
                prev_back_mask = selected
                missing_before_first = 0
            else:
                missing_before_first += 1
                if missing_before_first > self.config.max_missing_frames:
                    break

        return tracked

    def select_movable_instance_tokens(
        self,
        nusc: NuScenes,
        scene_names: list[str],
    ) -> dict[str, set[str]]:
        scene_map = {scene["name"]: scene for scene in nusc.scene}
        if not any(name in scene_map for name in scene_names):
            return {}

        sample_token_to_scene_name: dict[str, str] = {}
        for scene_name in scene_names:
            scene = scene_map.get(scene_name)
            if scene is None:
                continue
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = nusc.get("sample", sample_token)
                sample_token_to_scene_name[sample_token] = scene_name
                sample_token = sample.get("next", "")

        selected: dict[str, set[str]] = defaultdict(set)
        for ann in nusc.sample_annotation:
            sample_token = ann["sample_token"]
            scene_name = sample_token_to_scene_name.get(sample_token)
            if scene_name is None:
                continue

            visibility_token = int(ann.get("visibility_token", "4"))
            speed_kmh = self._safe_speed_kmh(nusc, ann["token"])
            if (
                speed_kmh >= self.config.speed_threshold_kmh
                or visibility_token <= self.config.visibility_token_max
            ):
                selected[scene_name].add(str(ann["instance_token"]))

        return selected

    def run(
        self,
        nusc: NuScenes,
        output_root: Path,
        scene_names: list[str],
        camera_names: list[str] | None = None,
    ) -> None:
        scene_map = {scene["name"]: scene for scene in nusc.scene}
        target_cameras = (
            [cam.upper() for cam in camera_names] if camera_names else DEFAULT_CAMERAS
        )

        scene_to_instances = self.select_movable_instance_tokens(nusc, scene_names)
        for scene_name in scene_names:
            scene = scene_map.get(scene_name)
            if scene is None:
                continue

            candidate_instances = sorted(scene_to_instances.get(scene_name, set()))
            if not candidate_instances:
                print(f"No candidate movable instances found in scene {scene_name}")
                continue

            for camera_name in target_cameras:
                cam_key = camera_name.lower()
                print(
                    f"Processing {scene_name} / {camera_name}: {len(candidate_instances)} candidate instances"
                )

                frame_infos = self._build_scene_camera_frames(
                    nusc=nusc,
                    scene_name=scene_name,
                    camera_name=camera_name,
                )
                if not frame_infos:
                    continue

                union_masks: dict[int, torch.Tensor] = {}
                progress_desc = (
                    f"{scene_name}/{cam_key} instances "
                    f"({len(candidate_instances)} total)"
                )
                for instance_token in tqdm(
                    candidate_instances,
                    desc=progress_desc,
                    unit="instance",
                    leave=False,
                ):
                    instance_masks = self._track_instance_for_camera(
                        nusc=nusc,
                        scene_token=scene["token"],
                        instance_token=instance_token,
                        frame_infos=frame_infos,
                    )
                    for frame_idx, inst_mask in instance_masks.items():
                        prior = union_masks.get(frame_idx)
                        if prior is None:
                            union_masks[frame_idx] = inst_mask.bool().to("cpu")
                        else:
                            union_masks[frame_idx] = torch.logical_or(
                                prior, inst_mask.bool().to("cpu")
                            )

                masked_frames: list[Image.Image] = []
                for frame_idx, frame in enumerate(frame_infos):
                    image = self._load_frame_image(nusc, frame)
                    width, height = image.size
                    layer_mask = union_masks.get(
                        frame_idx,
                        torch.zeros((height, width), dtype=torch.bool),
                    )
                    file_stem = f"{frame['timestamp']}_{cam_key}"
                    save_layer_mask(
                        output_root / scene_name / cam_key / "mask",
                        file_stem,
                        "movable_objects",
                        layer_mask,
                    )
                    masked_frames.append(apply_mask_to_image(image, layer_mask))

                video_path = (
                    output_root
                    / scene_name
                    / cam_key
                    / "visualization"
                    / f"{cam_key}_movable_layer_mask.mp4"
                )
                export_video_from_frames(masked_frames, video_path, fps=self.config.fps)
                print(f"Saved masked video for {scene_name}/{cam_key} to {video_path}")


def run_sam3_movable_object_preprocess(
    output_root: Path | None = None,
    scene_names: list[str] | None = None,
    camera_names: list[str] | None = None,
    nusc_version: str = "v1.0-mini",
    speed_threshold_kmh: float = 1.0,
    visibility_token_max: int = 2,
    max_missing_frames: int = 2,
    iou_threshold: float = 0.01,
    fps: int = 8,
) -> None:
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)

    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [scene["name"] for scene in nusc.scene]

    preprocessor = SAM3MovableObjectPreprocessor(
        SAM3MovableObjectPreprocessorConfig(
            speed_threshold_kmh=speed_threshold_kmh,
            visibility_token_max=visibility_token_max,
            max_missing_frames=max_missing_frames,
            iou_threshold=iou_threshold,
            fps=fps,
        )
    )
    preprocessor.run(
        nusc=nusc,
        output_root=resolve_dataset_root(output_root),
        scene_names=scene_names,
        camera_names=camera_names,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track NuScenes movable objects with SAM3 video preprocessor."
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
        "--nusc-version",
        default="v1.0-mini",
        help="NuScenes version (default: v1.0-mini)",
    )
    parser.add_argument(
        "--speed-threshold-kmh",
        type=float,
        default=1.0,
        help="Minimum speed in km/h to mark as movable candidate (default: 1.0)",
    )
    parser.add_argument(
        "--visibility-token-max",
        type=int,
        default=2,
        help="Maximum visibility token to mark as movable candidate (default: 2)",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=2,
        help="Stop propagation after this many consecutive empty frames (default: 2)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.01,
        help="Minimum IoU used for selecting per-instance masks (default: 0.01)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Output video FPS (default: 8)",
    )
    args = parser.parse_args()

    run_sam3_movable_object_preprocess(
        scene_names=args.scenes,
        camera_names=args.cameras,
        nusc_version=args.nusc_version,
        speed_threshold_kmh=args.speed_threshold_kmh,
        visibility_token_max=args.visibility_token_max,
        max_missing_frames=args.max_missing_frames,
        iou_threshold=args.iou_threshold,
        fps=args.fps,
    )

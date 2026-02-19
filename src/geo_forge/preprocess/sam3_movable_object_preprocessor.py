import argparse
import gc
from contextlib import nullcontext
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from nuscenes.nuscenes import NuScenes
from PIL import Image, ImageDraw
from pyquaternion import Quaternion
from dotenv import load_dotenv
from tqdm.auto import tqdm
from transformers import Sam3VideoModel, Sam3VideoProcessor
import yaml

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
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "sam3"
    / "movable_object_preprocessor"
    / "default.yaml"
)

load_dotenv()


@dataclass
class MovableObjectCandidateConfig:
    speed_threshold_kmh: float = 1.0
    visibility_token_max: int = 2
    always_include_category_prefixes: tuple[str, ...] = ("human.",)
    exclude_category_names: tuple[str, ...] = (
        "movable_object.barrier",
        "movable_object.trafficcone",
    )


@dataclass
class MovableObjectTrackingConfig:
    max_missing_frames: int = 2
    iou_threshold: float = 0.01
    max_center_distance_px: float = 120.0
    use_ego_yaw_mask_warp: bool = True
    yaw_warp_scale: float = 1.0
    reference_mask_dilation_kernel: int = 5
    reference_mask_dilation_iters: int = 1
    use_appearance_similarity: bool = False
    score_appearance_weight: float = 0.15
    use_model_confidence_score: bool = True
    score_model_confidence_weight: float = 0.2
    model_score_only_selection: bool = True
    use_sam3_score_top_k: bool = True
    sam3_score_top_k: int = 5
    use_area_stability_gate: bool = True
    min_area_retention_ratio: float = 0.25
    max_area_growth_ratio: float = 2.5
    score_area_stability_weight: float = 0.15
    retry_on_keyframe_gap_loss: bool = True
    retry_iou_threshold: float = 0.005
    retry_center_distance_scale: float = 2.0
    score_bbox_iou_weight: float = 0.5
    score_prev_iou_weight: float = 0.4
    score_center_distance_weight: float = 0.1
    keyframe_top_k_masks: int = 3
    min_bbox_overlap_ratio: float = 0.05


@dataclass
class MovableObjectOutputConfig:
    fps: int = 8


@dataclass
class SAM3MovableObjectPreprocessorConfig:
    model_name: str = "facebook/sam3"
    candidates: MovableObjectCandidateConfig = field(
        default_factory=MovableObjectCandidateConfig
    )
    tracking: MovableObjectTrackingConfig = field(
        default_factory=MovableObjectTrackingConfig
    )
    output: MovableObjectOutputConfig = field(default_factory=MovableObjectOutputConfig)

    @staticmethod
    def _ensure_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else ()
        if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
            normalized: list[str] = []
            for entry in value:
                text = str(entry).strip()
                if text:
                    normalized.append(text)
            return tuple(normalized)
        raise ValueError(f"{field_name} must be a string or list of strings")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SAM3MovableObjectPreprocessorConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")
        with config_path.open() as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"YAML at {config_path} must define a mapping.")

        model_name = str(raw.get("model_name", "facebook/sam3"))
        candidates_raw = raw.get("candidates") or {}
        tracking_raw = raw.get("tracking") or {}
        output_raw = raw.get("output") or {}
        if not isinstance(candidates_raw, dict):
            raise ValueError("candidates must be a mapping")
        if not isinstance(tracking_raw, dict):
            raise ValueError("tracking must be a mapping")
        if not isinstance(output_raw, dict):
            raise ValueError("output must be a mapping")

        candidates = MovableObjectCandidateConfig(
            speed_threshold_kmh=float(candidates_raw.get("speed_threshold_kmh", 1.0)),
            visibility_token_max=int(candidates_raw.get("visibility_token_max", 2)),
            always_include_category_prefixes=cls._ensure_str_tuple(
                candidates_raw.get("always_include_category_prefixes", ("human.",)),
                "always_include_category_prefixes",
            )
            or ("human.",),
            exclude_category_names=cls._ensure_str_tuple(
                candidates_raw.get(
                    "exclude_category_names",
                    ("movable_object.barrier", "movable_object.trafficcone"),
                ),
                "exclude_category_names",
            )
            or ("movable_object.barrier", "movable_object.trafficcone"),
        )
        tracking = MovableObjectTrackingConfig(
            max_missing_frames=int(tracking_raw.get("max_missing_frames", 2)),
            iou_threshold=float(tracking_raw.get("iou_threshold", 0.01)),
            max_center_distance_px=float(
                tracking_raw.get("max_center_distance_px", 120.0)
            ),
            use_ego_yaw_mask_warp=bool(tracking_raw.get("use_ego_yaw_mask_warp", True)),
            yaw_warp_scale=float(tracking_raw.get("yaw_warp_scale", 1.0)),
            reference_mask_dilation_kernel=int(
                tracking_raw.get("reference_mask_dilation_kernel", 5)
            ),
            reference_mask_dilation_iters=int(
                tracking_raw.get("reference_mask_dilation_iters", 1)
            ),
            use_appearance_similarity=bool(
                tracking_raw.get("use_appearance_similarity", False)
            ),
            score_appearance_weight=float(
                tracking_raw.get("score_appearance_weight", 0.15)
            ),
            use_model_confidence_score=bool(
                tracking_raw.get("use_model_confidence_score", True)
            ),
            score_model_confidence_weight=float(
                tracking_raw.get("score_model_confidence_weight", 0.2)
            ),
            model_score_only_selection=bool(
                tracking_raw.get("model_score_only_selection", True)
            ),
            use_sam3_score_top_k=bool(tracking_raw.get("use_sam3_score_top_k", True)),
            sam3_score_top_k=int(tracking_raw.get("sam3_score_top_k", 5)),
            use_area_stability_gate=bool(
                tracking_raw.get("use_area_stability_gate", True)
            ),
            min_area_retention_ratio=float(
                tracking_raw.get("min_area_retention_ratio", 0.25)
            ),
            max_area_growth_ratio=float(tracking_raw.get("max_area_growth_ratio", 2.5)),
            score_area_stability_weight=float(
                tracking_raw.get("score_area_stability_weight", 0.15)
            ),
            retry_on_keyframe_gap_loss=bool(
                tracking_raw.get("retry_on_keyframe_gap_loss", True)
            ),
            retry_iou_threshold=float(tracking_raw.get("retry_iou_threshold", 0.005)),
            retry_center_distance_scale=float(
                tracking_raw.get("retry_center_distance_scale", 2.0)
            ),
            score_bbox_iou_weight=float(tracking_raw.get("score_bbox_iou_weight", 0.5)),
            score_prev_iou_weight=float(tracking_raw.get("score_prev_iou_weight", 0.4)),
            score_center_distance_weight=float(
                tracking_raw.get("score_center_distance_weight", 0.1)
            ),
            keyframe_top_k_masks=int(tracking_raw.get("keyframe_top_k_masks", 3)),
            min_bbox_overlap_ratio=float(
                tracking_raw.get("min_bbox_overlap_ratio", 0.05)
            ),
        )
        output = MovableObjectOutputConfig(
            fps=int(output_raw.get("fps", 8)),
        )

        return cls(
            model_name=model_name,
            candidates=candidates,
            tracking=tracking,
            output=output,
        )


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
                else (torch.float16 if self.device.type == "cuda" else torch.float32)
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
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        with torch.inference_mode(), autocast_context:
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
        masks = masks.detach().to("cpu")

        boxes = processed.get("boxes")
        scores = processed.get("scores")
        labels = (
            processed.get("object_ids")
            if processed.get("object_ids") is not None
            else processed.get("labels")
        )
        boxes = boxes.detach().to("cpu") if boxes is not None else None
        scores = scores.detach().to("cpu") if scores is not None else None
        labels = labels.detach().to("cpu") if labels is not None else None

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
    def _category_to_prompt(category_name: str) -> str:
        category = category_name.strip().lower()
        if category.startswith("human."):
            return "human"
        if category.startswith("vehicle."):
            return "vehicle"
        if (
            category.startswith("cycle.")
            or "bicycle" in category
            or "motorcycle" in category
        ):
            return "bicycle"
        if category.startswith("animal."):
            return "animal"
        if "." in category:
            return category.split(".", maxsplit=1)[0].replace("_", " ")
        return category or "object"

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
    def _flatten_masks(
        mask_objects: list[ObjectMask],
    ) -> list[tuple[torch.Tensor, float]]:
        masks: list[tuple[torch.Tensor, float]] = []
        for mask_obj in mask_objects:
            if mask_obj.masks.numel() == 0:
                continue
            mask_t = mask_obj.masks.detach().to("cpu")
            score_t = (
                mask_obj.scores.detach().to("cpu")
                if isinstance(mask_obj.scores, torch.Tensor)
                else None
            )
            if mask_t.dim() == 2:
                score = (
                    float(score_t.reshape(-1)[0].item())
                    if score_t is not None and score_t.numel() > 0
                    else 1.0
                )
                masks.append((mask_t.bool(), score))
            elif mask_t.dim() == 3:
                for idx in range(mask_t.shape[0]):
                    score = (
                        float(score_t.reshape(-1)[idx].item())
                        if score_t is not None and score_t.numel() > idx
                        else 1.0
                    )
                    masks.append((mask_t[idx].bool(), score))
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

    @staticmethod
    def _mask_intersection(mask_a: torch.Tensor, mask_b: torch.Tensor) -> int:
        return int(torch.logical_and(mask_a, mask_b).sum().item())

    @staticmethod
    def _mask_centroid(mask: torch.Tensor) -> tuple[float, float] | None:
        ys, xs = torch.where(mask)
        if ys.numel() == 0:
            return None
        return (float(xs.float().mean().item()), float(ys.float().mean().item()))

    @staticmethod
    def _yaw_from_pose_quaternion(
        rotation_wxyz: list[float] | tuple[float, ...]
    ) -> float:
        return float(Quaternion(rotation_wxyz).yaw_pitch_roll[0])

    def _warp_and_expand_reference_mask(
        self,
        nusc: NuScenes,
        prev_frame: dict[str, Any] | None,
        current_frame: dict[str, Any],
        reference_mask: torch.Tensor | None,
        image_size: tuple[int, int],
    ) -> torch.Tensor | None:
        if reference_mask is None:
            return None

        width, height = image_size
        ref_mask = reference_mask.bool().to("cpu")
        if ref_mask.shape != (height, width):
            return ref_mask

        warped_mask = ref_mask
        if self.config.tracking.use_ego_yaw_mask_warp and prev_frame is not None:
            prev_pose = nusc.get("ego_pose", prev_frame["ego_pose_token"])
            curr_pose = nusc.get("ego_pose", current_frame["ego_pose_token"])
            prev_yaw = self._yaw_from_pose_quaternion(prev_pose["rotation"])
            curr_yaw = self._yaw_from_pose_quaternion(curr_pose["rotation"])
            yaw_delta_deg = math.degrees(
                (curr_yaw - prev_yaw) * float(self.config.tracking.yaw_warp_scale)
            )

            mask_img = Image.fromarray(
                (ref_mask.numpy().astype(np.uint8) * 255), mode="L"
            )
            warped_img = mask_img.rotate(
                angle=-yaw_delta_deg,
                resample=Image.Resampling.NEAREST,
                expand=False,
                fillcolor=0,
            )
            warped_mask = torch.from_numpy(np.asarray(warped_img, dtype=np.uint8) > 0)

        kernel = max(1, int(self.config.tracking.reference_mask_dilation_kernel))
        iters = max(0, int(self.config.tracking.reference_mask_dilation_iters))
        if kernel > 1 and iters > 0:
            pad = kernel // 2
            dil = warped_mask.float().unsqueeze(0).unsqueeze(0)
            for _ in range(iters):
                dil = F.max_pool2d(dil, kernel_size=kernel, stride=1, padding=pad)
            warped_mask = dil.squeeze(0).squeeze(0) > 0.5

        return warped_mask

    @staticmethod
    def _masked_rgb_embedding(
        image: Image.Image | None, mask: torch.Tensor | None
    ) -> np.ndarray | None:
        if image is None or mask is None:
            return None
        mask_np = mask.bool().to("cpu").numpy()
        img_np = np.asarray(image.convert("RGB"), dtype=np.float32)
        if mask_np.shape != img_np.shape[:2]:
            return None
        if not np.any(mask_np):
            return None
        pixels = img_np[mask_np]
        if pixels.size == 0:
            return None
        return pixels.mean(axis=0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-8:
            return 0.0
        value = float(np.dot(a, b) / denom)
        return float(max(-1.0, min(1.0, value)))

    def _select_best_mask(
        self,
        candidates: list[ObjectMask],
        image_size: tuple[int, int],
        bbox: tuple[float, float, float, float] | None = None,
        reference_mask: torch.Tensor | None = None,
        reference_image: Image.Image | None = None,
        current_image: Image.Image | None = None,
        prioritize_bbox: bool = False,
        allow_model_score_only: bool = True,
        iou_threshold: float | None = None,
        max_center_distance_px: float | None = None,
        min_bbox_overlap_ratio: float | None = None,
    ) -> torch.Tensor | None:
        has_model_scores = any(
            isinstance(mask_obj.scores, torch.Tensor) and mask_obj.scores.numel() > 0
            for mask_obj in candidates
        )
        flat = self._flatten_masks(candidates)
        if not flat:
            return None
        if has_model_scores and self.config.tracking.use_sam3_score_top_k:
            top_k = max(1, int(self.config.tracking.sam3_score_top_k))
            if len(flat) > top_k:
                flat = sorted(flat, key=lambda x: x[1], reverse=True)[:top_k]
        if (
            allow_model_score_only
            and has_model_scores
            and self.config.tracking.model_score_only_selection
        ):
            return max(flat, key=lambda x: x[1])[0]

        bbox_mask: torch.Tensor | None = None
        if bbox is not None:
            bbox_mask = self._bbox_to_mask(bbox, image_size)
        prev_mask = (
            reference_mask.bool().to("cpu") if reference_mask is not None else None
        )

        if bbox_mask is None and prev_mask is None:
            return flat[0][0]

        prev_center = self._mask_centroid(prev_mask) if prev_mask is not None else None
        iou_threshold_value = (
            float(self.config.tracking.iou_threshold)
            if iou_threshold is None
            else float(iou_threshold)
        )
        max_center_dist = (
            float(self.config.tracking.max_center_distance_px)
            if max_center_distance_px is None
            else float(max_center_distance_px)
        )
        min_bbox_overlap_ratio_value = (
            float(self.config.tracking.min_bbox_overlap_ratio)
            if min_bbox_overlap_ratio is None
            else float(min_bbox_overlap_ratio)
        )
        use_center_gate = prev_center is not None and max_center_dist > 0.0

        w_bbox = float(self.config.tracking.score_bbox_iou_weight)
        w_prev = float(self.config.tracking.score_prev_iou_weight)
        w_center = float(self.config.tracking.score_center_distance_weight)
        w_app = (
            float(self.config.tracking.score_appearance_weight)
            if self.config.tracking.use_appearance_similarity
            else 0.0
        )
        w_conf = (
            float(self.config.tracking.score_model_confidence_weight)
            if self.config.tracking.use_model_confidence_score
            else 0.0
        )
        w_area = float(self.config.tracking.score_area_stability_weight)
        if prioritize_bbox and bbox_mask is not None:
            w_bbox, w_prev, w_center, w_app = 0.75, 0.15, 0.05, min(0.05, w_app)
            w_conf = min(0.05, w_conf)
            w_area = min(0.05, w_area)

        reference_rgb = self._masked_rgb_embedding(reference_image, prev_mask)
        prev_area = int(prev_mask.sum().item()) if prev_mask is not None else 0
        min_retention = float(self.config.tracking.min_area_retention_ratio)
        max_growth = float(self.config.tracking.max_area_growth_ratio)

        scored_masks: list[tuple[float, float, torch.Tensor]] = []
        for cand, conf_score_raw in flat:
            if bbox_mask is not None and cand.shape != bbox_mask.shape:
                continue
            if prev_mask is not None and cand.shape != prev_mask.shape:
                continue

            bbox_iou = self._mask_iou(cand, bbox_mask) if bbox_mask is not None else 0.0
            prev_iou = self._mask_iou(cand, prev_mask) if prev_mask is not None else 0.0
            bbox_overlap_ratio = 0.0
            center_inside_bbox = False
            cand_area = int(cand.sum().item())
            if bbox_mask is not None:
                inter = self._mask_intersection(cand, bbox_mask)
                if cand_area > 0:
                    bbox_overlap_ratio = float(inter / cand_area)
                cand_center = self._mask_centroid(cand)
                if cand_center is not None:
                    cx, cy = cand_center
                    x_min, y_min, x_max, y_max = bbox
                    center_inside_bbox = x_min <= cx <= x_max and y_min <= cy <= y_max

            center_score = 0.0
            if prev_center is not None:
                cand_center = self._mask_centroid(cand)
                if cand_center is None:
                    continue
                dist = math.hypot(
                    cand_center[0] - prev_center[0], cand_center[1] - prev_center[1]
                )
                if use_center_gate and dist > max_center_dist:
                    continue
                center_score = 1.0 / (1.0 + dist)

            appearance_score = 0.0
            if w_app > 0.0 and current_image is not None:
                cand_rgb = self._masked_rgb_embedding(current_image, cand)
                # map cosine similarity [-1,1] -> [0,1]
                appearance_score = 0.5 * (
                    self._cosine_similarity(cand_rgb, reference_rgb) + 1.0
                )
            confidence_score = max(0.0, min(1.0, float(conf_score_raw)))
            area_score = 0.0
            if prev_area > 0 and cand_area > 0:
                area_ratio = float(cand_area / prev_area)
                if self.config.tracking.use_area_stability_gate and (
                    area_ratio < min_retention or area_ratio > max_growth
                ):
                    continue
                area_score = 1.0 / (1.0 + abs(math.log(area_ratio)))

            score = (
                w_bbox * bbox_iou
                + w_prev * prev_iou
                + w_center * center_score
                + w_app * appearance_score
                + w_conf * confidence_score
                + w_area * area_score
            )
            compat_iou = max(bbox_iou, prev_iou)
            if bbox_mask is not None:
                compat_iou = max(compat_iou, bbox_overlap_ratio)
                if center_inside_bbox:
                    compat_iou = max(compat_iou, iou_threshold_value)
            scored_masks.append((score, compat_iou, cand))

        if not scored_masks:
            return None
        scored_masks.sort(key=lambda x: x[0], reverse=True)

        best_score, best_compat_iou, best_mask = scored_masks[0]
        if best_mask is None or best_compat_iou < iou_threshold_value:
            return None

        if prioritize_bbox and bbox_mask is not None:
            top_k = max(1, int(self.config.tracking.keyframe_top_k_masks))
            selected_masks: list[torch.Tensor] = []
            for _, compat_iou, cand_mask in scored_masks[:top_k]:
                if compat_iou >= min(
                    iou_threshold_value,
                    min_bbox_overlap_ratio_value,
                ):
                    selected_masks.append(cand_mask.bool())
            if selected_masks:
                return torch.stack(selected_masks, dim=0).any(dim=0)

        return best_mask

    def _select_top1_mask_by_model_score(
        self,
        candidates: list[ObjectMask],
    ) -> torch.Tensor | None:
        flat = self._flatten_masks(candidates)
        if not flat:
            return None
        return max(flat, key=lambda x: x[1])[0]

    @staticmethod
    def _build_keyframe_gap_retry_indices(
        keyframe_indices: list[int],
    ) -> set[int]:
        retry_indices: set[int] = set()
        for left_idx, right_idx in zip(keyframe_indices, keyframe_indices[1:]):
            if right_idx - left_idx <= 1:
                continue
            for idx in range(left_idx + 1, right_idx):
                retry_indices.add(idx)
        return retry_indices

    def _select_best_mask_with_retry(
        self,
        *,
        candidates: list[ObjectMask],
        image_size: tuple[int, int],
        bbox: tuple[float, float, float, float] | None,
        reference_mask: torch.Tensor | None,
        reference_image: Image.Image | None,
        current_image: Image.Image | None,
        prioritize_bbox: bool,
        enable_relaxed_retry: bool,
        allow_model_score_only: bool = True,
    ) -> torch.Tensor | None:
        selected = self._select_best_mask(
            candidates=candidates,
            image_size=image_size,
            bbox=bbox,
            reference_mask=reference_mask,
            reference_image=reference_image,
            current_image=current_image,
            prioritize_bbox=prioritize_bbox,
            allow_model_score_only=allow_model_score_only,
        )
        if selected is not None or not enable_relaxed_retry:
            return selected

        relaxed_iou = min(
            float(self.config.tracking.iou_threshold),
            float(self.config.tracking.retry_iou_threshold),
        )
        relaxed_center_dist = float(self.config.tracking.max_center_distance_px) * max(
            1.0, float(self.config.tracking.retry_center_distance_scale)
        )
        return self._select_best_mask(
            candidates=candidates,
            image_size=image_size,
            bbox=bbox,
            reference_mask=reference_mask,
            reference_image=reference_image,
            current_image=current_image,
            prioritize_bbox=prioritize_bbox,
            allow_model_score_only=allow_model_score_only,
            iou_threshold=relaxed_iou,
            max_center_distance_px=relaxed_center_dist,
            min_bbox_overlap_ratio=relaxed_iou,
        )

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

    def _build_debug_bbox_index(
        self,
        nusc: NuScenes,
        scene_token: str,
        frame_infos: list[dict[str, Any]],
        instance_tokens: list[str],
    ) -> dict[int, list[tuple[str, tuple[float, float, float, float]]]]:
        keyframe_sizes: dict[int, tuple[int, int]] = {}
        for idx, frame in enumerate(frame_infos):
            if not frame["is_key_frame"]:
                continue
            image = self._load_frame_image(nusc, frame)
            keyframe_sizes[idx] = image.size

        bbox_index: dict[
            int, list[tuple[str, tuple[float, float, float, float]]]
        ] = defaultdict(list)
        for instance_token in instance_tokens:
            sample_to_ann = self._instance_sample_annotation_map(
                nusc=nusc, instance_token=instance_token, scene_token=scene_token
            )
            if not sample_to_ann:
                continue

            for idx, frame in enumerate(frame_infos):
                if not frame["is_key_frame"]:
                    continue
                sample_token = frame.get("sample_token")
                ann = sample_to_ann.get(sample_token)
                if ann is None:
                    continue

                image_size = keyframe_sizes.get(idx)
                if image_size is None:
                    continue

                box = NuscenesObjectBoundingBox.from_sample_annotation(ann)
                bbox = box.to_2d_bbox(
                    nusc=nusc,
                    calibrated_sensor_token=frame["calibrated_sensor_token"],
                    ego_pose_token=frame["ego_pose_token"],
                    image_size=image_size,
                )
                if bbox is None:
                    continue
                bbox_index[idx].append((instance_token, bbox))
        return dict(bbox_index)

    @staticmethod
    def _draw_instance_bboxes(
        image: Image.Image,
        bboxes: list[tuple[str, tuple[float, float, float, float]]],
    ) -> Image.Image:
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        for instance_token, (x_min, y_min, x_max, y_max) in bboxes:
            draw.rectangle(
                [int(x_min), int(y_min), int(x_max), int(y_max)],
                outline=(255, 64, 64),
                width=3,
            )
            draw.text(
                (int(x_min) + 2, max(0, int(y_min) - 14)),
                instance_token[:8],
                fill=(255, 220, 64),
            )
        return vis

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
        frame_idx_to_category: dict[int, str] = {}

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
            frame_idx_to_category[idx] = str(ann.get("category_name", "object"))

        if not frame_idx_to_bbox:
            return {}

        first_idx = min(frame_idx_to_bbox.keys())
        last_idx = max(frame_idx_to_bbox.keys())
        prompt = self._category_to_prompt(frame_idx_to_category[first_idx])
        keyframe_indices = sorted(frame_idx_to_bbox.keys())
        retry_indices = self._build_keyframe_gap_retry_indices(keyframe_indices)

        tracked: dict[int, torch.Tensor] = {}
        session: Any | None = None
        session_back: Any | None = None
        try:
            # Forward tracking: keyframe interval + post-disappearance sweeps.
            prev_mask: torch.Tensor | None = None
            prev_image: Image.Image | None = None
            missing_after_last = 0
            after_disappearance_keyframe = False
            for idx in range(first_idx, len(frame_infos)):
                frame = frame_infos[idx]
                if session is None or (idx != first_idx and frame["is_key_frame"]):
                    self._release_inference_session(session)
                    session = self.init_streaming_session(prompt)
                image = self._load_frame_image(nusc, frame)
                prev_frame = frame_infos[idx - 1] if idx > first_idx else None
                reference_mask = self._warp_and_expand_reference_mask(
                    nusc=nusc,
                    prev_frame=prev_frame,
                    current_frame=frame,
                    reference_mask=prev_mask,
                    image_size=image.size,
                )
                candidates = self.stream_video_frame(session, image, reverse=False)
                if frame["is_key_frame"]:
                    if idx in frame_idx_to_bbox:
                        selected = self._select_best_mask_with_retry(
                            candidates=candidates,
                            image_size=image.size,
                            bbox=frame_idx_to_bbox.get(idx),
                            reference_mask=reference_mask,
                            reference_image=prev_image,
                            current_image=image,
                            prioritize_bbox=True,
                            enable_relaxed_retry=(
                                self.config.tracking.retry_on_keyframe_gap_loss
                                and idx in retry_indices
                            ),
                            allow_model_score_only=False,
                        )
                    else:
                        selected = None
                        if idx > last_idx:
                            after_disappearance_keyframe = True
                else:
                    selected = self._select_top1_mask_by_model_score(candidates)

                if selected is not None:
                    tracked[idx] = selected
                    prev_mask = selected
                    prev_image = image
                    missing_after_last = 0
                else:
                    if frame["is_key_frame"] and idx not in frame_idx_to_bbox:
                        prev_mask = None
                        prev_image = None
                    if after_disappearance_keyframe:
                        missing_after_last += 1
                        if missing_after_last > self.config.tracking.max_missing_frames:
                            break
                    elif idx > last_idx:
                        missing_after_last += 1
                        if missing_after_last > self.config.tracking.max_missing_frames:
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
                reference_image=None,
                current_image=anchor_image,
                prioritize_bbox=True,
                allow_model_score_only=False,
            )

            prev_back_mask = anchor_mask
            prev_back_image: Image.Image | None = anchor_image
            if anchor_mask is not None:
                tracked[first_idx] = anchor_mask

            missing_before_first = 0
            before_appearance_keyframe = False
            for idx in range(first_idx - 1, -1, -1):
                frame = frame_infos[idx]
                if frame["is_key_frame"]:
                    self._release_inference_session(session_back)
                    session_back = self.init_streaming_session(prompt)
                image = self._load_frame_image(nusc, frame)
                prev_frame = (
                    frame_infos[idx + 1] if idx + 1 < len(frame_infos) else None
                )
                reference_mask = self._warp_and_expand_reference_mask(
                    nusc=nusc,
                    prev_frame=prev_frame,
                    current_frame=frame,
                    reference_mask=prev_back_mask,
                    image_size=image.size,
                )
                candidates = self.stream_video_frame(session_back, image, reverse=True)
                if frame["is_key_frame"]:
                    if idx in frame_idx_to_bbox:
                        selected = self._select_best_mask_with_retry(
                            candidates=candidates,
                            image_size=image.size,
                            bbox=frame_idx_to_bbox.get(idx),
                            reference_mask=reference_mask,
                            reference_image=prev_back_image,
                            current_image=image,
                            prioritize_bbox=True,
                            enable_relaxed_retry=(
                                self.config.tracking.retry_on_keyframe_gap_loss
                                and idx in retry_indices
                            ),
                            allow_model_score_only=False,
                        )
                    else:
                        selected = None
                        if idx < first_idx:
                            before_appearance_keyframe = True
                else:
                    selected = self._select_top1_mask_by_model_score(candidates)

                if selected is not None:
                    tracked[idx] = selected
                    prev_back_mask = selected
                    prev_back_image = image
                    missing_before_first = 0
                else:
                    if frame["is_key_frame"] and idx not in frame_idx_to_bbox:
                        prev_back_mask = None
                        prev_back_image = None
                    missing_before_first += 1
                    if missing_before_first > self.config.tracking.max_missing_frames:
                        break
                    if before_appearance_keyframe:
                        break

            return tracked
        finally:
            self._release_inference_session(session_back)
            self._release_inference_session(session)

    def _release_inference_session(self, session: Any | None) -> None:
        if session is None:
            return

        for method_name in (
            "close_video_session",
            "reset_video_session",
            "clear_video_session",
            "release_video_session",
        ):
            method = getattr(self.processor, method_name, None)
            if not callable(method):
                continue
            try:
                method(session)
                break
            except TypeError:
                try:
                    method(inference_session=session)
                    break
                except TypeError:
                    continue

        del session
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

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

            category_name = str(ann.get("category_name", ""))
            if category_name in self.config.candidates.exclude_category_names:
                continue

            if any(
                category_name.startswith(prefix)
                for prefix in self.config.candidates.always_include_category_prefixes
            ):
                selected[scene_name].add(str(ann["instance_token"]))
                continue

            visibility_token = int(ann.get("visibility_token", "4"))
            speed_kmh = self._safe_speed_kmh(nusc, ann["token"])
            if (
                speed_kmh >= self.config.candidates.speed_threshold_kmh
                or visibility_token <= self.config.candidates.visibility_token_max
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

                debug_bbox_index = self._build_debug_bbox_index(
                    nusc=nusc,
                    scene_token=scene["token"],
                    frame_infos=frame_infos,
                    instance_tokens=candidate_instances,
                )
                instance_bbox_frames: list[Image.Image] = []
                for frame_idx, frame in enumerate(frame_infos):
                    image = self._load_frame_image(nusc, frame)
                    frame_bboxes = (
                        debug_bbox_index.get(frame_idx, [])
                        if frame["is_key_frame"]
                        else []
                    )
                    instance_bbox_frames.append(
                        self._draw_instance_bboxes(image, frame_bboxes)
                    )

                instance_video_path = (
                    output_root
                    / scene_name
                    / cam_key
                    / "visualization"
                    / f"{cam_key}_movable_object_instance.mp4"
                )
                export_video_from_frames(
                    instance_bbox_frames,
                    instance_video_path,
                    fps=self.config.output.fps,
                )
                print(
                    f"Saved instance bbox video for {scene_name}/{cam_key} to {instance_video_path}"
                )

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
                export_video_from_frames(
                    masked_frames, video_path, fps=self.config.output.fps
                )
                print(f"Saved masked video for {scene_name}/{cam_key} to {video_path}")


def run_sam3_movable_object_preprocess(
    config_path: Path | str | None = None,
    config: SAM3MovableObjectPreprocessorConfig | None = None,
    output_root: Path | None = None,
    scene_names: list[str] | None = None,
    camera_names: list[str] | None = None,
    nusc_version: str = "v1.0-mini",
    speed_threshold_kmh: float | None = None,
    visibility_token_max: int | None = None,
    max_missing_frames: int | None = None,
    iou_threshold: float | None = None,
    fps: int | None = None,
    always_include_category_prefixes: tuple[str, ...] | None = None,
    exclude_category_names: tuple[str, ...] | None = None,
) -> None:
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version=nusc_version, dataroot=dataroot, verbose=True)

    if scene_names is None:
        if not nusc.scene:
            raise ValueError("NuScenes dataset is empty")
        scene_names = [scene["name"] for scene in nusc.scene]

    if config is None:
        resolved_config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        config = SAM3MovableObjectPreprocessorConfig.from_yaml(resolved_config_path)

    if speed_threshold_kmh is not None:
        config.candidates.speed_threshold_kmh = float(speed_threshold_kmh)
    if visibility_token_max is not None:
        config.candidates.visibility_token_max = int(visibility_token_max)
    if max_missing_frames is not None:
        config.tracking.max_missing_frames = int(max_missing_frames)
    if iou_threshold is not None:
        config.tracking.iou_threshold = float(iou_threshold)
    if fps is not None:
        config.output.fps = int(fps)
    if always_include_category_prefixes is not None:
        config.candidates.always_include_category_prefixes = tuple(
            always_include_category_prefixes
        )
    if exclude_category_names is not None:
        config.candidates.exclude_category_names = tuple(exclude_category_names)

    preprocessor = SAM3MovableObjectPreprocessor(config)
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
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH}).",
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
        default=None,
        help="Override YAML: minimum speed in km/h to mark as movable candidate.",
    )
    parser.add_argument(
        "--visibility-token-max",
        type=int,
        default=None,
        help="Override YAML: maximum visibility token to mark as movable candidate.",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=None,
        help="Override YAML: stop propagation after this many consecutive empty frames.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Override YAML: minimum IoU used for selecting per-instance masks.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override YAML: output video FPS.",
    )
    parser.add_argument(
        "--always-include-category-prefix",
        action="append",
        dest="always_include_category_prefixes",
        help=("Category prefix to always include (repeatable). " "Default: human."),
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        dest="exclude_category_names",
        help=(
            "Category name to always exclude (repeatable). "
            "Defaults: movable_object.barrier, movable_object.trafficcone"
        ),
    )
    args = parser.parse_args()

    run_sam3_movable_object_preprocess(
        config_path=args.config,
        scene_names=args.scenes,
        camera_names=args.cameras,
        nusc_version=args.nusc_version,
        speed_threshold_kmh=args.speed_threshold_kmh,
        visibility_token_max=args.visibility_token_max,
        max_missing_frames=args.max_missing_frames,
        iou_threshold=args.iou_threshold,
        fps=args.fps,
        always_include_category_prefixes=tuple(args.always_include_category_prefixes)
        if args.always_include_category_prefixes
        else None,
        exclude_category_names=tuple(args.exclude_category_names)
        if args.exclude_category_names
        else None,
    )

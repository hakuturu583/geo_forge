from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from rich import print as rprint
from transformers import AutoTokenizer
import transformers.utils as _hf_utils
from dotenv import load_dotenv
from nuscenes.nuscenes import NuScenes

# diffusers expects FLAX constants present in older transformers releases; provide fallbacks.
if not hasattr(_hf_utils, "FLAX_WEIGHTS_NAME"):
    _hf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"  # type: ignore[attr-defined]

from diffusers import FlowMatchEulerDiscreteScheduler

# Load environment variables from a local .env if present (for ROSE paths, etc.).
load_dotenv()

from geo_forge.dataclass import ObjectMask
from geo_forge.preprocess.preprocess import export_video_from_frames
from rose.models import (
    AutoencoderKLWan,
    CLIPModel,
    WanT5EncoderModel,
    WanTransformer3DModel,
)
from rose.pipeline import WanFunInpaintPipeline
from rose.utils.utils import color_transfer


def _filter_kwargs(cls: type, kwargs: dict) -> dict:
    """Strip kwargs that are not accepted by the target class constructor."""
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


class RosePreprocessor:
    """
    Thin wrapper around the ROSE inpainting pipeline.

    Consumes a list of RGB frames and a list of ``ObjectMask`` instances
    (aligned per-frame), runs ROSE to remove the masked regions, and returns the
    resulting video as a list of ``PIL.Image`` frames.
    """

    def __init__(
        self,
        model_root: str = "models/Wan2.1-Fun-1.3B-InP",
        transformer_root: str = "weights/transformer",
        model_repo_id: str | None = None,
        transformer_repo_id: str | None = None,
        config_path: str | Path = "configs/wan2.1/wan_civitai.yaml",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        default_inference_steps: int = 50,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = (
            dtype
            if dtype is not None
            else (torch.float16 if self.device.type == "cuda" else torch.float32)
        )
        self.default_inference_steps = default_inference_steps
        self.model_repo_id = model_repo_id or os.getenv("ROSE_MODEL_REPO_ID")
        self.transformer_repo_id = transformer_repo_id or os.getenv(
            "ROSE_TRANSFORMER_REPO_ID"
        )

        # If repo IDs are not provided but the roots are not local paths, treat roots as repo IDs.
        if self.model_repo_id is None and not os.path.exists(model_root):
            self.model_repo_id = model_root
        if self.transformer_repo_id is None and not os.path.exists(transformer_root):
            self.transformer_repo_id = transformer_root
        self.pipeline = self._load_pipeline(
            model_root=model_root,
            transformer_root=transformer_root,
            config_path=config_path,
        ).to(self.device, self.dtype)

    def _load_pipeline(
        self, model_root: str, transformer_root: str, config_path: str
    ) -> WanFunInpaintPipeline:
        config_path = str(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"ROSE config not found at {config_path}. "
                "Set ROSE_CONFIG_PATH to a valid config and ensure weights are available locally."
            )

        config = OmegaConf.load(config_path)
        tokenizer_subpath = config["text_encoder_kwargs"].get(
            "tokenizer_subpath", "tokenizer"
        )
        text_encoder_subpath = config["text_encoder_kwargs"].get(
            "text_encoder_subpath", "text_encoder"
        )
        image_encoder_subpath = config["image_encoder_kwargs"].get(
            "image_encoder_subpath", "image_encoder"
        )
        vae_subpath = config["vae_kwargs"].get("vae_subpath", "vae")
        transformer_subpath = config["transformer_additional_kwargs"].get(
            "transformer_subpath", "transformer"
        )

        tokenizer_local_path = os.path.join(model_root, tokenizer_subpath)
        if os.path.exists(tokenizer_local_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path)
        elif "/" in tokenizer_subpath:
            rprint(
                f"[yellow]Tokenizer path not found locally; loading tokenizer from repo id '{tokenizer_subpath}'.[/yellow]"
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_subpath)
        else:
            tokenizer_path = self._resolve_path_or_hub(
                base=model_root,
                subpath=tokenizer_subpath,
                repo_id=self.model_repo_id,
                local_label="Tokenizer",
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        text_encoder = WanT5EncoderModel.from_pretrained(
            self._resolve_path_or_hub(
                base=model_root,
                subpath=text_encoder_subpath,
                repo_id=self.model_repo_id,
                local_label="Text encoder",
            ),
            additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
            low_cpu_mem_usage=True,
        )

        clip_image_encoder = CLIPModel.from_pretrained(
            self._resolve_path_or_hub(
                base=model_root,
                subpath=image_encoder_subpath,
                repo_id=self.model_repo_id,
                local_label="Image encoder",
            ),
        )

        scheduler = FlowMatchEulerDiscreteScheduler(
            **_filter_kwargs(
                FlowMatchEulerDiscreteScheduler,
                OmegaConf.to_container(config["scheduler_kwargs"]),
            )
        )

        vae = AutoencoderKLWan.from_pretrained(
            self._resolve_path_or_hub(
                base=model_root,
                subpath=vae_subpath,
                repo_id=self.model_repo_id,
                local_label="VAE",
            ),
            additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
        )

        transformer3d = WanTransformer3DModel.from_pretrained(
            self._resolve_path_or_hub(
                base=transformer_root,
                subpath=transformer_subpath,
                repo_id=self.transformer_repo_id,
                local_label="Transformer",
            ),
            transformer_additional_kwargs=OmegaConf.to_container(
                config["transformer_additional_kwargs"]
            ),
        )

        return WanFunInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer3d,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )

    @staticmethod
    def _object_mask_to_bool(
        mask_obj: ObjectMask, expected_hw: tuple[int, int]
    ) -> torch.Tensor:
        if mask_obj.masks.numel() == 0:
            return torch.zeros(expected_hw, dtype=torch.bool)

        mask_tensor = mask_obj.masks.detach().to("cpu")
        if mask_tensor.dim() == 3:
            mask = mask_tensor.any(dim=0)
        elif mask_tensor.dim() == 2:
            mask = mask_tensor.bool()
        else:
            raise ValueError(f"Unsupported mask dimensionality: {mask_tensor.dim()}")

        if mask.shape != expected_hw:
            raise ValueError(
                "Mask and image spatial dimensions do not match: "
                f"mask={mask.shape[::-1]}, image={(expected_hw[1], expected_hw[0])}"
            )
        return mask

    @staticmethod
    def _to_pil_video(
        videos: torch.Tensor,
        rescale: bool = False,
        n_rows: int = 6,
        color_transfer_post_process: bool = False,
    ) -> List[Image.Image]:
        videos = rearrange(videos, "b c t h w -> t b c h w")
        outputs: List[Image.Image] = []
        for frame_batch in videos:
            grid = torchvision.utils.make_grid(frame_batch, nrow=n_rows)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            if rescale:
                grid = (grid + 1.0) / 2.0
            grid = (grid * 255).numpy().astype(np.uint8)
            outputs.append(Image.fromarray(grid))

        if color_transfer_post_process and outputs:
            for i in range(1, len(outputs)):
                outputs[i] = Image.fromarray(
                    color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0]))
                )

        return outputs

    def remove_objects(
        self,
        frames: Sequence[Image.Image],
        masks: Sequence[ObjectMask],
        prompt: str = "",
        num_inference_steps: int | None = None,
        color_transfer_post_process: bool = False,
        scale: float = 0.5,
        mask_dilation: int = 1,
        guidance_scale: float = 6.0,
        frame_batch_size: int = 33,
        batch_output_dir: Path | None = None,
        batch_export_fps: int = 12,
        frame_output_dir: Path | None = None,
        frame_output_stems: Sequence[str] | None = None,
        camera_name: str | None = None,
    ) -> List[Image.Image]:
        """
        Run ROSE inpainting to remove masked objects from a video.

        Args:
            frames: Video frames as PIL images (all frames must share dimensions).
            masks: Per-frame ``ObjectMask`` predictions aligned with ``frames``.
            prompt: Optional text prompt passed to the ROSE pipeline.
            num_inference_steps: Override the default diffusion steps.
            color_transfer_post_process: Whether to harmonize colors using the first frame.
            scale: Spatial downscale factor applied before ROSE (restored after inference).
            mask_dilation: Kernel size for mask dilation (pixels); must be > 0.
            frame_batch_size: Number of frames to process per ROSE call (limits VRAM use).
            batch_output_dir: Optional directory to write per-batch GIFs for debugging/monitoring.
            batch_export_fps: Frame rate for per-batch GIF exports.
            frame_output_dir: Optional directory to save per-frame outputs as batches complete.
            frame_output_stems: Filenames (sans extension) for per-frame outputs; must align with ``frames``.
            camera_name: Camera name appended to saved filenames when ``frame_output_dir`` is provided.

        Returns:
            List of inpainted frames as ``PIL.Image`` objects.
        """
        if not frames:
            raise ValueError("frames is empty; expected at least one frame.")
        if len(frames) != len(masks):
            raise ValueError(
                f"frames and masks must align one-to-one (got {len(frames)} frames, {len(masks)} masks)."
            )

        if scale <= 0:
            raise ValueError(f"scale must be positive (got {scale})")
        if mask_dilation <= 0:
            raise ValueError(f"mask_dilation must be positive (got {mask_dilation})")
        if frame_batch_size <= 0:
            raise ValueError(
                f"frame_batch_size must be positive (got {frame_batch_size})"
            )
        if batch_export_fps <= 0:
            raise ValueError(
                f"batch_export_fps must be positive (got {batch_export_fps})"
            )

        orig_width, orig_height = frames[0].size
        for frame in frames:
            if frame.size != (orig_width, orig_height):
                raise ValueError(
                    "All frames must share dimensions for ROSE inpainting."
                )

        stems: list[str] | None = None
        frame_output_dir_path: Path | None = None
        if frame_output_dir is not None:
            if frame_output_stems is None:
                raise ValueError(
                    "frame_output_stems must be provided when frame_output_dir is set."
                )
            if len(frame_output_stems) != len(frames):
                raise ValueError(
                    "frame_output_stems must align with frames when saving outputs "
                    f"(got {len(frame_output_stems)} stems for {len(frames)} frames)."
                )
            stems = list(frame_output_stems)
            frame_output_dir_path = Path(frame_output_dir)
            frame_output_dir_path.mkdir(parents=True, exist_ok=True)

        batched_outputs: list[Image.Image] = []
        for batch_idx, start in enumerate(range(0, len(frames), frame_batch_size)):
            end = start + frame_batch_size
            batch_frames = frames[start:end]
            batch_masks = masks[start:end]
            batch_stems = stems[start:end] if stems is not None else None
            batch_output = self._remove_objects_batch(
                batch_frames,
                batch_masks,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                color_transfer_post_process=color_transfer_post_process,
                scale=scale,
                mask_dilation=mask_dilation,
                guidance_scale=guidance_scale,
                orig_size=(orig_width, orig_height),
            )
            batched_outputs.extend(batch_output)

            if frame_output_dir_path is not None and batch_stems is not None:
                if len(batch_stems) != len(batch_output):
                    raise RuntimeError(
                        "Batch output length does not match provided stems "
                        f"({len(batch_output)} outputs vs {len(batch_stems)} stems)."
                    )
                for frame, stem in zip(batch_output, batch_stems):
                    filename = (
                        f"{stem}_{camera_name}.png"
                        if camera_name is not None
                        else f"{stem}.png"
                    )
                    frame.save(frame_output_dir_path / filename)

            if batch_output_dir is not None:
                batch_output_path = (
                    Path(batch_output_dir) / f"batch_{batch_idx:03d}.gif"
                )
                export_video_from_frames(
                    batch_output, batch_output_path, fps=batch_export_fps
                )
        return batched_outputs

    def _remove_objects_batch(
        self,
        frames: Sequence[Image.Image],
        masks: Sequence[ObjectMask],
        *,
        prompt: str,
        num_inference_steps: int | None,
        color_transfer_post_process: bool,
        scale: float,
        mask_dilation: int,
        guidance_scale: float,
        orig_size: tuple[int, int],
    ) -> List[Image.Image]:
        if not frames:
            raise ValueError("frames is empty; expected at least one frame.")
        if len(frames) != len(masks):
            raise ValueError(
                f"frames and masks must align one-to-one (got {len(frames)} frames, {len(masks)} masks)."
            )

        orig_width, orig_height = orig_size

        # Validate masks against original size, then resize both frames and masks.
        masks_bool_orig = [
            self._object_mask_to_bool(mask_obj, (orig_height, orig_width))
            for mask_obj in masks
        ]

        scaled_width = max(8, int((orig_width * scale) // 8 * 8))
        scaled_height = max(8, int((orig_height * scale) // 8 * 8))
        width, height = scaled_width, scaled_height

        resized_frames: list[Image.Image] = []
        for frame in frames:
            if frame.size != (width, height):
                resized_frames.append(frame.resize((width, height), Image.BICUBIC))
            else:
                resized_frames.append(frame)

        resized_masks: list[torch.Tensor] = []
        for mask_bool in masks_bool_orig:
            mask_img = Image.fromarray(mask_bool.cpu().numpy().astype(np.uint8) * 255)
            if mask_img.size != (width, height):
                mask_img = mask_img.resize((width, height), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_img) > 0)
            if mask_dilation > 1:
                mask_f = mask_tensor.unsqueeze(0).unsqueeze(0).float()
                pad = mask_dilation // 2
                dilated = F.max_pool2d(
                    mask_f, kernel_size=mask_dilation, stride=1, padding=pad
                )
                if dilated.shape[-2:] != (height, width):
                    dilated = dilated[..., :height, :width]
                mask_tensor = dilated.squeeze(0).squeeze(0) > 0
            resized_masks.append(mask_tensor)

        original_frame_count = len(resized_frames)

        # ROSE pipeline expects (num_frames % 4 == 1) after its internal conditioning;
        # pad with the last frame/mask to satisfy this constraint.
        pad_count = (1 - len(resized_frames) % 4) % 4
        if pad_count:
            resized_frames = list(resized_frames) + [resized_frames[-1]] * pad_count
            resized_masks = list(resized_masks) + [resized_masks[-1]] * pad_count

        mask_tensor = torch.stack(resized_masks, dim=0).unsqueeze(1).unsqueeze(0)
        mask_tensor = mask_tensor.permute(0, 2, 1, 3, 4).float()

        frame_tensors: List[torch.Tensor] = []
        for frame in resized_frames:
            frame_arr = np.asarray(frame.convert("RGB")).copy()
            frame_tensor = torch.from_numpy(frame_arr).permute(2, 0, 1)
            frame_tensors.append(frame_tensor)
        video_tensor = torch.stack(frame_tensors, dim=1).unsqueeze(0).float() / 255.0

        result = self.pipeline(
            prompt=prompt,
            video=video_tensor.to(self.device, self.dtype),
            mask_video=mask_tensor.to(self.device, self.dtype),
            height=height,
            width=width,
            num_frames=video_tensor.shape[2],
            num_inference_steps=num_inference_steps or self.default_inference_steps,
            guidance_scale=guidance_scale,
            return_dict=False,
        ).videos

        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result)

        if result.dim() != 5:
            raise ValueError(f"Unexpected video output shape from ROSE: {result.shape}")

        pil_frames = self._to_pil_video(
            result,
            rescale=False,
            n_rows=1,
            color_transfer_post_process=color_transfer_post_process,
        )
        if pad_count:
            pil_frames = pil_frames[:original_frame_count]

        # Restore to original spatial resolution.
        if (width, height) != (orig_width, orig_height):
            pil_frames = [
                frame.resize((orig_width, orig_height), Image.BICUBIC)
                for frame in pil_frames
            ]
        return pil_frames

    def _resolve_path_or_hub(
        self, base: str, subpath: str, repo_id: str | None, local_label: str
    ) -> str:
        """Resolve a weight path, downloading from HF Hub if not found locally."""
        local_path = os.path.join(base, subpath)
        if os.path.exists(local_path):
            return local_path

        if repo_id:
            if subpath in {"", ".", "./"}:
                rprint(
                    f"[yellow]{local_label} not found locally; fetching full snapshot from {repo_id}[/yellow]"
                )
                return snapshot_download(repo_id=repo_id)
            rprint(
                f"[yellow]{local_label} not found locally; fetching {subpath} from {repo_id}[/yellow]"
            )
            return hf_hub_download(repo_id=repo_id, filename=subpath)

        raise FileNotFoundError(
            f"{local_label} missing: {local_path}. "
            "Provide ROSE_MODEL_REPO_ID/ROSE_TRANSFORMER_REPO_ID or download locally."
        )


def _build_frame_index(
    nusc: NuScenes,
    scene_filter: set[str] | None = None,
    camera_filter: set[str] | None = None,
) -> dict[tuple[str, str, int], str]:
    """
    Build a lookup from (scene_name, camera, timestamp) to NuScenes image paths.

    Includes both keyframes (indexed by LiDAR and camera timestamps) and
    intermediate camera sweeps (indexed by camera timestamps).
    """
    scene_name_by_token = {scene["token"]: scene["name"] for scene in nusc.scene}
    if scene_filter is None:
        allowed_scene_tokens = set(scene_name_by_token.keys())
    else:
        allowed_scene_tokens = {
            token for token, name in scene_name_by_token.items() if name in scene_filter
        }

    sample_token_to_scene: dict[str, str] = {}
    for sample in nusc.sample:
        if sample["scene_token"] not in allowed_scene_tokens:
            continue
        sample_token_to_scene[sample["token"]] = scene_name_by_token[
            sample["scene_token"]
        ]

    sample_token_to_lidar_ts: dict[str, int] = {}
    for sample_data in nusc.sample_data:
        if not sample_data.get("is_key_frame", True):
            continue
        if sample_data.get("channel") != "LIDAR_TOP":
            continue
        if sample_data["sample_token"] not in sample_token_to_scene:
            continue
        sample_token_to_lidar_ts[sample_data["sample_token"]] = int(
            sample_data["timestamp"]
        )

    frame_index: dict[tuple[str, str, int], str] = {}
    for sample_data in nusc.sample_data:
        if not sample_data.get("channel", "").startswith("CAM_"):
            continue

        sample_token = sample_data["sample_token"]
        scene_name = sample_token_to_scene.get(sample_token)
        if scene_name is None:
            continue

        lidar_ts = sample_token_to_lidar_ts.get(sample_token)

        channel = sample_data.get("channel", "").lower()
        if camera_filter and channel not in camera_filter:
            continue

        timestamp = int(sample_data["timestamp"])
        frame_index[(scene_name, channel, timestamp)] = sample_data["filename"]

        # Also index keyframes by LiDAR timestamp to match legacy mask stems.
        if sample_data.get("is_key_frame", True) and lidar_ts is not None:
            frame_index.setdefault(
                (scene_name, channel, lidar_ts), sample_data["filename"]
            )

    return frame_index


def _load_scene_frames_and_masks(
    cam_dir: Path,
    frame_index: dict[tuple[str, str, int], str],
    dataroot: Path,
    enforce_16n_plus_one: bool = False,
) -> tuple[list[Image.Image], list[ObjectMask], list[str]]:
    """
    Load raw NuScenes frames (by timestamp) and movable-object masks.

    Args:
        cam_dir: Path to a camera directory containing masks.
        frame_index: Lookup of (scene_name, camera, timestamp) -> relative image path
            (supports LiDAR timestamps for keyframes and camera timestamps for sweeps).
        dataroot: Root of the NuScenes dataset (NUSCENES_DATAROOT).
        enforce_16n_plus_one: If True, truncate to the largest sequence length of
            the form 16n+1 (may return empty lists when no masks are present).

    Returns:
        Frames, mask objects, and the original file stems (sans the mask suffix).
    """
    scene_name = cam_dir.parent.name
    cam_name = cam_dir.name
    mask_root = cam_dir / "mask"
    mask_paths = sorted(mask_root.glob("*_movable_objects.pt"))
    if not mask_paths:
        raise FileNotFoundError(f"No movable_objects masks found under {mask_root}")

    if enforce_16n_plus_one:
        target_count = ((len(mask_paths) - 1) // 16) * 16 + 1 if mask_paths else 0
        mask_paths = mask_paths[:target_count]
        if target_count <= 0:
            return [], [], []

    frames: list[Image.Image] = []
    mask_objects: list[ObjectMask] = []
    mask_stems: list[str] = []
    for mask_path in mask_paths:
        mask_tensor = torch.load(mask_path, map_location="cpu")
        mask_objects.append(ObjectMask(masks=mask_tensor.bool()))

        stem = mask_path.stem
        if stem.endswith("_movable_objects"):
            stem = stem.removesuffix("_movable_objects")
        mask_stems.append(stem)

        try:
            timestamp_str, _ = stem.split("_", 1)
            timestamp = int(timestamp_str)
        except ValueError:
            raise ValueError(
                f"Mask stem '{stem}' does not start with a timestamp; expected '<timestamp>_{cam_name}'."
            )

        frame_rel = frame_index.get((scene_name, cam_name.lower(), timestamp))
        if frame_rel is None:
            raise FileNotFoundError(
                f"No NuScenes frame found for scene={scene_name}, camera={cam_name}, timestamp={timestamp}. "
                "Ensure masks were generated from the same dataset."
            )

        frame_path = dataroot / frame_rel
        if not frame_path.exists():
            raise FileNotFoundError(f"NuScenes frame not found at {frame_path}")

        with Image.open(frame_path) as img:
            frames.append(img.convert("RGB").copy())

    if len(frames) != len(mask_paths):
        raise ValueError(
            f"Frame/mask length mismatch: {len(frames)} frames vs {len(mask_paths)} masks"
        )

    return frames, mask_objects, mask_stems


def _concat_frames_side_by_side(
    left_frames: Sequence[Image.Image], right_frames: Sequence[Image.Image]
) -> list[Image.Image]:
    """
    Join pairs of frames horizontally for visualization.
    """
    if len(left_frames) != len(right_frames):
        raise ValueError(
            f"Cannot concatenate sequences of different lengths: {len(left_frames)} vs {len(right_frames)}"
        )

    concatenated: list[Image.Image] = []
    for left, right in zip(left_frames, right_frames):
        if left.size != right.size:
            raise ValueError(
                f"Frame size mismatch: left={left.size}, right={right.size}"
            )
        width, height = left.size
        canvas = Image.new("RGB", (width * 2, height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (width, 0))
        concatenated.append(canvas)
    return concatenated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ROSE object removal over preprocessed scenes."
    )
    parser.add_argument(
        "--scene",
        "-s",
        action="append",
        dest="scenes",
        help="Scene directory name to process (repeatable). Defaults to all scenes.",
    )
    parser.add_argument(
        "--camera",
        "-c",
        action="append",
        dest="cameras",
        help="Camera directory name to process (repeatable). Defaults to all cameras.",
    )
    parser.add_argument(
        "--enforce-16n-plus-1",
        action="store_true",
        dest="enforce_16n_plus_one",
        help=(
            "Truncate frames/masks to the largest length of the form 16n+1; "
            "if none remain, the scene/camera is skipped."
        ),
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=33,
        help=(
            "Process frames in batches to limit VRAM usage (number of frames per ROSE call)."
        ),
    )
    args = parser.parse_args()
    scene_filter = set(args.scenes) if args.scenes else None
    camera_filter = {cam.lower() for cam in args.cameras} if args.cameras else None

    dataset_root = Path(__file__).resolve().parent / "datasets"
    dataroot = Path(os.getenv("NUSCENES_DATAROOT", "/data/nuscenes"))
    nusc_version = os.getenv("NUSCENES_VERSION", "v1.0-mini")
    model_root = os.getenv("ROSE_MODEL_ROOT", "models/Wan2.1-Fun-1.3B-InP")
    transformer_root = os.getenv("ROSE_TRANSFORMER_ROOT", "weights/transformer")
    config_path = os.getenv("ROSE_CONFIG_PATH", "configs/wan2.1/wan_civitai.yaml")

    if not Path(config_path).exists():
        rprint(
            f"[red]Config not found at {config_path}. Set ROSE_CONFIG_PATH to your ROSE config and ensure weights are downloaded.[/red]"
        )
        raise SystemExit(1)

    if not dataroot.exists():
        rprint(
            f"[red]NuScenes dataroot not found at {dataroot}. Set NUSCENES_DATAROOT to your dataset path.[/red]"
        )
        raise SystemExit(1)

    nusc = NuScenes(version=nusc_version, dataroot=str(dataroot), verbose=False)
    frame_index = _build_frame_index(nusc, scene_filter, camera_filter)

    preprocessor = RosePreprocessor(
        model_root=model_root,
        transformer_root=transformer_root,
        config_path=config_path,
        default_inference_steps=100,
    )

    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if scene_filter and scene_dir.name not in scene_filter:
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            if camera_filter and cam_dir.name not in camera_filter:
                continue
            try:
                frames, masks, mask_stems = _load_scene_frames_and_masks(
                    cam_dir,
                    frame_index,
                    dataroot,
                    enforce_16n_plus_one=args.enforce_16n_plus_one,
                )
            except FileNotFoundError as e:
                rprint(f"[yellow]Skipping {cam_dir}: {e}[/yellow]")
                continue
            if args.enforce_16n_plus_one and not frames:
                rprint(
                    f"[yellow]Skipping {cam_dir}: no frames available after enforcing 16n+1 length.[/yellow]"
                )
                continue
            visualization_root = cam_dir / "visualization"
            batch_visualization_root = visualization_root / "rose_object_removal"
            output_path = visualization_root / "object_removed.gif"
            images_output_dir = (
                cam_dir / "object_removed_images" / "object_removed_images"
            )
            visualization_root.mkdir(parents=True, exist_ok=True)
            # input_gif_path = visualization_root / f"{cam_dir.name}_input.gif"
            # export_video_from_frames(frames, input_gif_path, fps=12)
            inpainted_frames = preprocessor.remove_objects(
                frames,
                masks,
                prompt="",
                color_transfer_post_process=False,
                mask_dilation=9,
                frame_batch_size=args.frame_batch_size,
                batch_output_dir=batch_visualization_root,
                batch_export_fps=12,
                frame_output_dir=images_output_dir,
                frame_output_stems=mask_stems,
                camera_name=cam_dir.name,
            )
            export_video_from_frames(inpainted_frames, output_path, fps=12)
            comparison_frames = _concat_frames_side_by_side(frames, inpainted_frames)
            comparison_gif_path = (
                visualization_root / f"{cam_dir.name}_input_vs_object_removed.gif"
            )
            export_video_from_frames(comparison_frames, comparison_gif_path, fps=12)
            # print(f"Saved input GIF to {input_gif_path}")
            print(f"Saved object-removed video to {output_path}")
            print(f"Saved side-by-side GIF to {comparison_gif_path}")
            print(f"Saved object-removed frames to {images_output_dir}")

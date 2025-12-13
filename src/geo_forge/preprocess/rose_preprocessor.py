from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import List, Sequence

import imageio.v3 as iio
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

# diffusers expects FLAX constants present in older transformers releases; provide fallbacks.
if not hasattr(_hf_utils, "FLAX_WEIGHTS_NAME"):
    _hf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"  # type: ignore[attr-defined]

from diffusers import FlowMatchEulerDiscreteScheduler

# Load environment variables from a local .env if present (for ROSE paths, etc.).
load_dotenv()

from geo_forge.dataclass import ObjectMask
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

        orig_width, orig_height = frames[0].size
        for frame in frames:
            if frame.size != (orig_width, orig_height):
                raise ValueError(
                    "All frames must share dimensions for ROSE inpainting."
                )

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


def _load_scene_frames_and_masks(
    cam_dir: Path,
) -> tuple[list[Image.Image], list[ObjectMask], list[str]]:
    """
    Load frames and movable-object masks from a preprocessed scene directory.

    Args:
        cam_dir: Path to a camera directory containing visualization GIFs and masks.

    Returns:
        Frames, mask objects, and the original file stems (sans the mask suffix).
    """
    cam_name = cam_dir.name
    gif_path = cam_dir / "visualization" / f"{cam_name}_raw.gif"
    mask_root = cam_dir / "mask"
    mask_paths = sorted(mask_root.glob("*_movable_objects.pt"))
    if not gif_path.exists():
        raise FileNotFoundError(f"Raw frames GIF not found at {gif_path}")
    if not mask_paths:
        raise FileNotFoundError(f"No movable_objects masks found under {mask_root}")

    frames_np = iio.imread(gif_path)
    frames = [Image.fromarray(frame) for frame in frames_np]
    if len(frames) != len(mask_paths):
        raise ValueError(
            f"Frame/mask length mismatch: {len(frames)} frames vs {len(mask_paths)} masks"
        )
    mask_objects: list[ObjectMask] = []
    mask_stems: list[str] = []
    for mask_path in mask_paths:
        mask_tensor = torch.load(mask_path, map_location="cpu")
        mask_objects.append(ObjectMask(masks=mask_tensor.bool()))

        stem = mask_path.stem
        if stem.endswith("_movable_objects"):
            stem = stem.removesuffix("_movable_objects")
        mask_stems.append(stem)

    return frames, mask_objects, mask_stems


if __name__ == "__main__":
    dataset_root = Path(__file__).resolve().parent / "datasets"
    model_root = os.getenv("ROSE_MODEL_ROOT", "models/Wan2.1-Fun-1.3B-InP")
    transformer_root = os.getenv("ROSE_TRANSFORMER_ROOT", "weights/transformer")
    config_path = os.getenv("ROSE_CONFIG_PATH", "configs/wan2.1/wan_civitai.yaml")

    if not Path(config_path).exists():
        rprint(
            f"[red]Config not found at {config_path}. Set ROSE_CONFIG_PATH to your ROSE config and ensure weights are downloaded.[/red]"
        )
        raise SystemExit(1)

    preprocessor = RosePreprocessor(
        model_root=model_root,
        transformer_root=transformer_root,
        config_path=config_path,
    )

    from geo_forge.preprocess.preprocess import export_video_from_frames

    for scene_dir in sorted(dataset_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for cam_dir in sorted(scene_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            try:
                frames, masks, mask_stems = _load_scene_frames_and_masks(cam_dir)
            except FileNotFoundError as e:
                rprint(f"[yellow]Skipping {cam_dir}: {e}[/yellow]")
                continue
            visualization_root = cam_dir / "visualization"
            output_path = visualization_root / f"{cam_dir.name}_object_removed.gif"
            images_output_dir = visualization_root / "object_removed_images"
            inpainted_frames = preprocessor.remove_objects(
                frames,
                masks,
                prompt="",
                color_transfer_post_process=False,
                mask_dilation=9,
            )
            export_video_from_frames(inpainted_frames, output_path, fps=12)
            images_output_dir.mkdir(parents=True, exist_ok=True)
            if len(inpainted_frames) != len(mask_stems):
                raise RuntimeError(
                    f"Mismatch between output frames ({len(inpainted_frames)}) and mask stems ({len(mask_stems)})"
                )
            for frame, stem in zip(inpainted_frames, mask_stems):
                frame.save(images_output_dir / f"{stem}_object_removed.png")
            print(f"Saved object-removed video to {output_path}")
            print(f"Saved object-removed frames to {images_output_dir}")

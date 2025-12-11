from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from diffusers.utils import load_image
from pipeline_difix import DifixPipeline


class DifixPreprocessor:
    """
    Thin wrapper around Difix for batch image cleanup.

    Looks for ``object_removed_images`` under a camera directory and writes the
    cleaned frames to ``cleanuped_images``.
    """

    def __init__(
        self,
        model_id: str = "nvidia/difix",
        device: str | torch.device | None = None,
        trust_remote_code: bool = True,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.pipeline = DifixPipeline.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.pipeline.to(self.device)

    def cleanup_camera_directory(
        self,
        camera_dir: str | Path,
        *,
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: Sequence[int] | None = None,
        guidance_scale: float = 0.0,
        overwrite: bool = False,
    ) -> List[Path]:
        """
        Run Difix over all frames in ``object_removed_images`` for a camera.

        Args:
            camera_dir: Path to a camera directory containing ``object_removed_images``.
            prompt: Text prompt passed to Difix.
            num_inference_steps: Scheduler steps for inference.
            timesteps: Optional list of timesteps; defaults to ``[199]`` to mirror the demo script.
            guidance_scale: Guidance scale forwarded to Difix.
            overwrite: Whether to regenerate frames when output already exists.

        Returns:
            List of output image paths created or reused.
        """
        camera_path = Path(camera_dir)
        input_dir = camera_path / "object_removed_images"
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir = camera_path / "cleanuped_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestep_list: Sequence[int] = timesteps if timesteps is not None else [199]
        image_paths = sorted(
            p
            for p in input_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )

        outputs: List[Path] = []
        for image_path in image_paths:
            output_path = output_dir / image_path.name
            if output_path.exists() and not overwrite:
                outputs.append(output_path)
                continue

            input_image = load_image(str(image_path))
            result = self.pipeline(
                prompt,
                image=input_image,
                num_inference_steps=num_inference_steps,
                timesteps=timestep_list,
                guidance_scale=guidance_scale,
            )
            cleaned_image = result.images[0]
            cleaned_image.save(output_path)
            outputs.append(output_path)

        return outputs

    def cleanup_dataset(
        self,
        dataset_root: str | Path,
        *,
        scene_names: Iterable[str] | None = None,
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: Sequence[int] | None = None,
        guidance_scale: float = 0.0,
        overwrite: bool = False,
    ) -> Dict[Path, List[Path]]:
        """
        Traverse scenes/cameras under a dataset root and clean frames.

        Returns a mapping of camera directories to the written output paths.
        """
        root = Path(dataset_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        target_scenes: Iterable[Path]
        if scene_names:
            selected = []
            for name in scene_names:
                scene_path = root / name
                if scene_path.is_dir():
                    selected.append(scene_path)
            target_scenes = selected
        else:
            target_scenes = [p for p in root.iterdir() if p.is_dir()]

        results: Dict[Path, List[Path]] = {}
        for scene_dir in target_scenes:
            for camera_dir in scene_dir.iterdir():
                if not camera_dir.is_dir():
                    continue
                input_dir = camera_dir / "object_removed_images"
                if not input_dir.is_dir():
                    continue
                outputs = self.cleanup_camera_directory(
                    camera_dir,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    overwrite=overwrite,
                )
                results[camera_dir] = outputs
        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Difix over object_removed_images to produce cleanuped_images."
    )
    default_root = Path(__file__).resolve().parent / "datasets"
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help=f"Root containing scene folders (default: {default_root})",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional list of scene directory names to process.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/difix",
        help="Model id or local path for Difix.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for the pipeline (defaults to cuda if available).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="remove degradation",
        help="Prompt passed to Difix.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help="Number of inference steps for Difix.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[199],
        help="Timesteps forwarded to Difix (space separated).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Guidance scale forwarded to Difix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs even if cleanuped_images already exists.",
    )
    args = parser.parse_args()

    preprocessor = DifixPreprocessor(
        model_id=args.model_id,
        device=args.device,
    )
    results = preprocessor.cleanup_dataset(
        dataset_root=args.dataset_root,
        scene_names=args.scenes,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        timesteps=args.timesteps,
        guidance_scale=args.guidance_scale,
        overwrite=args.overwrite,
    )

    for camera_dir, outputs in sorted(results.items()):
        print(f"{camera_dir}: wrote {len(outputs)} frame(s)")


if __name__ == "__main__":
    main()

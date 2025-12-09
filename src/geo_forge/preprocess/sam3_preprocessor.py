"""Generic SAM3 video preprocessor for object tracking and segmentation"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import Image
import cv2
from abc import ABC, abstractmethod

from transformers import AutoModel, AutoProcessor
from ..dataclass import SAM3PreprocessorConfig


class VideoDataAdapter(ABC):
    """Abstract base class for video data adapters"""

    @abstractmethod
    def get_frames(self, identifier: Any, **kwargs) -> List[Image.Image]:
        """Get video frames for a given identifier"""
        pass

    @abstractmethod
    def get_prompts(
        self, identifier: Any, frame_idx: int = 0, **kwargs
    ) -> Dict[int, Union[List[List[float]], Dict]]:
        """Get object prompts for initialization"""
        pass

    @abstractmethod
    def get_metadata(self, identifier: Any, **kwargs) -> Dict[str, Any]:
        """Get metadata for the video sequence"""
        pass


class SAM3Preprocessor:
    """Generic SAM3 preprocessor for video tracking and segmentation"""

    def __init__(
        self,
        config: Optional[SAM3PreprocessorConfig] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_masks: Optional[bool] = None,
        save_visualizations: Optional[bool] = None,
        verbose: Optional[bool] = None,
    ):
        """
        Initialize SAM3 preprocessor

        Args:
            config: SAM3PreprocessorConfig instance, uses default if None
            model_id: HuggingFace model ID for SAM2 (overrides config)
            device: Device to use (cuda/cpu), auto-detect if None (overrides config)
            output_dir: Directory to save outputs (overrides config)
            save_masks: Whether to save mask arrays (overrides config)
            save_visualizations: Whether to save visualization images (overrides config)
            verbose: Whether to print progress (overrides config)
        """
        # Use config or create default
        self.config = config or SAM3PreprocessorConfig()

        # Override config values with explicit parameters if provided
        self.model_id = model_id or self.config.model_id
        self.device = self._setup_device(device or self.config.device)
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.save_masks = (
            save_masks if save_masks is not None else self.config.save_masks
        )
        self.save_visualizations = (
            save_visualizations
            if save_visualizations is not None
            else self.config.save_visualizations
        )
        self.verbose = verbose if verbose is not None else self.config.verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and processor
        self._init_model()

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup computation device"""
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self):
        """Initialize SAM model and processor"""
        if self.verbose:
            print(f"Loading model: {self.model_id}")

        # Use auto classes so SAM2/SAM3 remote code is supported without hard dependencies
        processor_kwargs = {"trust_remote_code": True, "local_files_only": True}
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, **processor_kwargs
            )
        except Exception:
            # Fall back to allow fetching missing artifacts if cache is incomplete
            processor_kwargs.pop("local_files_only", None)
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, **processor_kwargs
            )

        model_kwargs = {"trust_remote_code": True, "local_files_only": True}
        try:
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        except Exception:
            model_kwargs.pop("local_files_only", None)
            self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.verbose:
            print(f"Model loaded on device: {self.device}")

    def process_video(
        self,
        frames: List[Image.Image],
        prompts: Optional[Dict[int, Any]] = None,
        prompt_texts: Optional[List[str]] = None,
        identifier: str = "video",
        mask_threshold: Optional[float] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        points_per_batch: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process video frames with SAM3 for tracking and segmentation.

        Args:
            frames: List of PIL images.
            prompts: Deprecated; kept for compatibility but unused.
            prompt_texts: Text prompts for detection (defaults to config.target_classes).
            identifier: Identifier for saving results.
            mask_threshold: Threshold for binary mask generation (uses config default if None).
            pred_iou_thresh: Predicted IoU threshold for filtering masks (unused in SAM3 flow).
            stability_score_thresh: Stability score threshold (unused in SAM3 flow).
            points_per_batch: Points per batch for processing (kept for API parity).
            **kwargs: Additional arguments for processor.

        Returns:
            Dictionary containing processing results.
        """
        if not frames:
            raise ValueError("No frames provided")

        # Use config defaults if not specified
        mask_threshold = (
            mask_threshold if mask_threshold is not None else self.config.mask_threshold
        )
        pred_iou_thresh = (
            pred_iou_thresh
            if pred_iou_thresh is not None
            else self.config.pred_iou_thresh
        )
        stability_score_thresh = (
            stability_score_thresh
            if stability_score_thresh is not None
            else self.config.stability_score_thresh
        )
        points_per_batch = (
            points_per_batch
            if points_per_batch is not None
            else self.config.points_per_batch
        )

        if self.verbose:
            print(f"Processing {len(frames)} frames with identifier: {identifier}")

        prompt_texts = prompt_texts or self.config.target_classes
        if not prompt_texts:
            raise ValueError("No prompt_texts provided for SAM3 processing.")

        # Prepare video session and add text prompts
        inference_session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            inference_state_device=self.device,
            processing_device=self.device,
            video_storage_device=self.device,
        )
        for text_prompt in prompt_texts:
            self.processor.add_text_prompt(inference_session, text_prompt)

        # Generate masks frame by frame
        if self.verbose:
            print("Generating masks...")

        masks_by_frame: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for output in self.model.propagate_in_video_iterator(
                inference_session,
                max_frame_num_to_track=len(frames),
            ):
                frame_idx = output.frame_idx
                obj_id_to_mask = output.obj_id_to_mask or {}
                if not obj_id_to_mask:
                    continue

                upsampled_masks = []
                for mask in obj_id_to_mask.values():
                    if mask is None:
                        continue
                    # Convert logits to probabilities and optionally threshold before upsampling
                    mask_probs = torch.sigmoid(mask.float())
                    if mask_threshold is not None:
                        mask_probs = (mask_probs > mask_threshold).float()
                    upsampled = torch.nn.functional.interpolate(
                        mask_probs.unsqueeze(0),
                        size=(inference_session.video_height, inference_session.video_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    upsampled_masks.append(upsampled)

                if upsampled_masks:
                    masks_by_frame[frame_idx] = torch.stack(upsampled_masks)

        # Order masks to align with frame list
        masks = [masks_by_frame.get(i, torch.tensor([])) for i in range(len(frames))]

        # Save results
        results = self._save_results(identifier, frames, masks)

        if self.verbose:
            print(f"Processing complete. Results saved to: {results['output_dir']}")

        return results

    def process_with_adapter(
        self, adapter: VideoDataAdapter, identifier: Any, **process_kwargs
    ) -> Dict[str, Any]:
        """
        Process video using a data adapter

        Args:
            adapter: VideoDataAdapter instance
            identifier: Identifier for the video sequence
            **process_kwargs: Additional arguments for process_video

        Returns:
            Dictionary containing processing results
        """
        # Get frames from adapter
        frames = adapter.get_frames(identifier)

        # Get prompts from adapter
        prompts = adapter.get_prompts(identifier, frame_idx=0)

        # Get metadata
        metadata = adapter.get_metadata(identifier)

        # Process video
        results = self.process_video(
            frames=frames, prompts=prompts, identifier=str(identifier), **process_kwargs
        )

        # Add metadata to results
        results["metadata"] = metadata

        return results

    def _save_results(
        self, identifier: str, frames: List[Image.Image], masks: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Save processing results

        Args:
            identifier: Identifier for the results
            frames: Original video frames
            masks: Generated masks

        Returns:
            Dictionary with save paths and statistics
        """
        save_dir = self.output_dir / identifier
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "identifier": identifier,
            "num_frames": len(frames),
            "num_masks": sum(
                mask.shape[0]
                for mask in masks
                if isinstance(mask, torch.Tensor) and mask.ndim > 0
            ),
            "output_dir": str(save_dir),
            "mask_files": [],
            "visualization_files": [],
        }

        if self.save_masks:
            masks_dir = save_dir / "masks"
            masks_dir.mkdir(exist_ok=True)

            for idx, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                    mask_file = masks_dir / f"mask_{idx:04d}.npy"
                    np.save(mask_file, mask_np)
                    results["mask_files"].append(str(mask_file))

        if self.save_visualizations:
            vis_dir = save_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            for idx, (frame, mask) in enumerate(zip(frames, masks)):
                vis_file = vis_dir / f"frame_{idx:04d}.jpg"
                self._save_visualization(frame, mask, vis_file)
                results["visualization_files"].append(str(vis_file))

        return results

    def _save_visualization(
        self, frame: Image.Image, mask: torch.Tensor, output_path: Path
    ):
        """
        Save visualization of frame with mask overlay

        Args:
            frame: Original frame
            mask: Mask tensor
            output_path: Path to save visualization
        """
        frame_np = np.array(frame)

        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()

            # Create colored overlay
            overlay = frame_np.copy()

            if mask_np.size == 0:
                # Nothing to draw; save the raw frame
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
                return

            if mask_np.ndim == 3:  # Multiple objects
                colors = self._generate_colors(mask_np.shape[0])
                for obj_idx in range(mask_np.shape[0]):
                    mask_binary = mask_np[obj_idx] > 0.5
                    overlay[mask_binary] = (
                        overlay[mask_binary] * 0.5 + colors[obj_idx] * 0.5
                    )
            elif mask_np.ndim == 2:  # Single mask
                mask_binary = mask_np > 0.5
                overlay[mask_binary] = (
                    overlay[mask_binary] * 0.5 + np.array([0, 255, 0]) * 0.5
                )
            else:
                # Unsupported shape; save raw frame
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
                return

            # Save visualization
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

    def _generate_colors(self, n: int) -> np.ndarray:
        """Generate distinct colors for n objects"""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(n):
            hue = i / n
            # Convert HSV to RGB
            rgb = np.array([hue * 360, 1.0, 1.0])  # HSV
            # Simple HSV to RGB conversion (approximation)
            colors.append(
                [
                    int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + i / 3))))
                    for i in range(3)
                ]
            )
        return np.array(colors)


def example_usage():
    """Example usage with simple frames"""

    # Create sample frames (replace with actual video frames)
    frames = []
    for i in range(5):
        # Create dummy frame
        frame = Image.new("RGB", (640, 480), color=(100 + i * 20, 100, 100))
        frames.append(frame)

    # Create text prompts for tracking
    prompt_texts = ["car", "pedestrian"]

    # Initialize preprocessor
    preprocessor = SAM3Preprocessor(
        model_id="facebook/sam3", output_dir="./sam3_test_output", verbose=True
    )

    # Process video
    results = preprocessor.process_video(
        frames=frames, prompt_texts=prompt_texts, identifier="test_video"
    )

    print(f"\nResults:")
    print(f"  Output directory: {results['output_dir']}")
    print(f"  Frames processed: {results['num_frames']}")
    print(f"  Masks generated: {results['num_masks']}")


if __name__ == "__main__":
    example_usage()

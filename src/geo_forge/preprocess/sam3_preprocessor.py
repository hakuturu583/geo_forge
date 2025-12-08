"""Generic SAM3 video preprocessor for object tracking and segmentation"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import Image
import cv2
from abc import ABC, abstractmethod

from transformers import SAM2VideoProcessor, SAM2ForUniversalSegmentation
from ..dataclass import SAM3PreprocessorConfig


class VideoDataAdapter(ABC):
    """Abstract base class for video data adapters"""
    
    @abstractmethod
    def get_frames(self, identifier: Any, **kwargs) -> List[Image.Image]:
        """Get video frames for a given identifier"""
        pass
    
    @abstractmethod
    def get_prompts(self, identifier: Any, frame_idx: int = 0, **kwargs) -> Dict[int, Union[List[List[float]], Dict]]:
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
        verbose: Optional[bool] = None
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
        self.save_masks = save_masks if save_masks is not None else self.config.save_masks
        self.save_visualizations = save_visualizations if save_visualizations is not None else self.config.save_visualizations
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
        
        self.processor = SAM2VideoProcessor.from_pretrained(self.model_id)
        self.model = SAM2ForUniversalSegmentation.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.verbose:
            print(f"Model loaded on device: {self.device}")
    
    def process_video(
        self,
        frames: List[Image.Image],
        prompts: Dict[int, Any],
        identifier: str = "video",
        mask_threshold: Optional[float] = None,
        pred_iou_thresh: Optional[float] = None,
        stability_score_thresh: Optional[float] = None,
        points_per_batch: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process video frames with SAM3 for tracking and segmentation
        
        Args:
            frames: List of PIL images
            prompts: Dictionary mapping frame indices to prompts (points or boxes)
            identifier: Identifier for saving results
            mask_threshold: Threshold for binary mask generation (uses config default if None)
            pred_iou_thresh: Predicted IoU threshold for filtering masks (uses config default if None)
            stability_score_thresh: Stability score threshold (uses config default if None)
            points_per_batch: Points per batch for processing (uses config default if None)
            **kwargs: Additional arguments for processor
            
        Returns:
            Dictionary containing processing results
        """
        if not frames:
            raise ValueError("No frames provided")
        
        # Use config defaults if not specified
        mask_threshold = mask_threshold if mask_threshold is not None else self.config.mask_threshold
        pred_iou_thresh = pred_iou_thresh if pred_iou_thresh is not None else self.config.pred_iou_thresh
        stability_score_thresh = stability_score_thresh if stability_score_thresh is not None else self.config.stability_score_thresh
        points_per_batch = points_per_batch if points_per_batch is not None else self.config.points_per_batch
        
        if self.verbose:
            print(f"Processing {len(frames)} frames with identifier: {identifier}")
        
        # Process with SAM
        inputs = self.processor(
            images=frames,
            points=prompts if isinstance(list(prompts.values())[0], list) else None,
            boxes=prompts if isinstance(list(prompts.values())[0], dict) else None,
            return_tensors="pt",
            points_per_batch=points_per_batch,
            **kwargs
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate masks
        if self.verbose:
            print("Generating masks...")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process masks
        masks = self.processor.post_process_video_segmentation(
            outputs,
            original_sizes=[img.size[::-1] for img in frames],
            mask_threshold=mask_threshold,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        
        # Save results
        results = self._save_results(identifier, frames, masks)
        
        if self.verbose:
            print(f"Processing complete. Results saved to: {results['output_dir']}")
        
        return results
    
    def process_with_adapter(
        self,
        adapter: VideoDataAdapter,
        identifier: Any,
        **process_kwargs
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
            frames=frames,
            prompts=prompts,
            identifier=str(identifier),
            **process_kwargs
        )
        
        # Add metadata to results
        results['metadata'] = metadata
        
        return results
    
    def _save_results(
        self,
        identifier: str,
        frames: List[Image.Image],
        masks: List[torch.Tensor]
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
            'identifier': identifier,
            'num_frames': len(frames),
            'num_masks': len(masks),
            'output_dir': str(save_dir),
            'mask_files': [],
            'visualization_files': []
        }
        
        if self.save_masks:
            masks_dir = save_dir / 'masks'
            masks_dir.mkdir(exist_ok=True)
            
            for idx, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                    mask_file = masks_dir / f'mask_{idx:04d}.npy'
                    np.save(mask_file, mask_np)
                    results['mask_files'].append(str(mask_file))
        
        if self.save_visualizations:
            vis_dir = save_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for idx, (frame, mask) in enumerate(zip(frames, masks)):
                vis_file = vis_dir / f'frame_{idx:04d}.jpg'
                self._save_visualization(frame, mask, vis_file)
                results['visualization_files'].append(str(vis_file))
        
        return results
    
    def _save_visualization(
        self,
        frame: Image.Image,
        mask: torch.Tensor,
        output_path: Path
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
            
            if mask_np.ndim == 3:  # Multiple objects
                colors = self._generate_colors(mask_np.shape[0])
                for obj_idx in range(mask_np.shape[0]):
                    mask_binary = mask_np[obj_idx] > 0.5
                    overlay[mask_binary] = overlay[mask_binary] * 0.5 + colors[obj_idx] * 0.5
            else:  # Single mask
                mask_binary = mask_np > 0.5
                overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([0, 255, 0]) * 0.5
            
            # Save visualization
            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
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
            colors.append([
                int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + i/3))))
                for i in range(3)
            ])
        return np.array(colors)


def example_usage():
    """Example usage with simple frames"""
    
    # Create sample frames (replace with actual video frames)
    frames = []
    for i in range(5):
        # Create dummy frame
        frame = Image.new('RGB', (640, 480), color=(100 + i*20, 100, 100))
        frames.append(frame)
    
    # Create sample prompts (center points)
    prompts = {
        0: [[320, 240], [200, 300]]  # Two points in first frame
    }
    
    # Initialize preprocessor
    preprocessor = SAM3Preprocessor(
        model_id="facebook/sam2-hiera-large",
        output_dir="./sam3_test_output",
        verbose=True
    )
    
    # Process video
    results = preprocessor.process_video(
        frames=frames,
        prompts=prompts,
        identifier="test_video"
    )
    
    print(f"\nResults:")
    print(f"  Output directory: {results['output_dir']}")
    print(f"  Frames processed: {results['num_frames']}")
    print(f"  Masks generated: {results['num_masks']}")


if __name__ == "__main__":
    example_usage()
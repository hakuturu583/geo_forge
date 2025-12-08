"""SAM3 video tracking for NuScenes dataset"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import cv2

from transformers import SAM2VideoProcessor, SAM2ForUniversalSegmentation
from nuscenes.nuscenes import NuScenes
from dotenv import load_dotenv

from .dataclass import SAM3DPreprocessorConfig
from .nuscenes import iterate_synchronized_samples, load_synchronized_data

# Load environment variables
load_dotenv()


class NuScenesSAM3Tracker:
    """SAM3 video tracking for NuScenes scenes"""
    
    def __init__(self, config: Optional[SAM3DPreprocessorConfig] = None):
        """
        Initialize SAM3 tracker with configuration
        
        Args:
            config: SAM3DPreprocessorConfig instance, uses default if None
        """
        self.config = config or SAM3DPreprocessorConfig()
        
        # Initialize SAM3 model and processor
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.processor = SAM2VideoProcessor.from_pretrained(self.config.model_id)
        self.model = SAM2ForUniversalSegmentation.from_pretrained(self.config.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize NuScenes
        dataroot = os.getenv('NUSCENES_DATAROOT', '/data/nuscenes')
        self.nusc = NuScenes(
            version=self.config.version,
            dataroot=dataroot,
            verbose=self.config.verbose
        )
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_scene_frames(self, scene_name: str, camera: str = 'CAM_FRONT') -> List[Dict[str, Any]]:
        """
        Get all synchronized frames for a specific scene and camera
        
        Args:
            scene_name: Name of the NuScenes scene
            camera: Camera to use (default CAM_FRONT)
            
        Returns:
            List of frame information dictionaries
        """
        frames = []
        for sample_info in iterate_synchronized_samples(self.nusc, scene_names=[scene_name], cameras=[camera]):
            if camera in sample_info['cameras']:
                frames.append(sample_info)
                if self.config.max_frames_per_scene and len(frames) >= self.config.max_frames_per_scene:
                    break
        return frames
    
    def load_frames_as_video(self, frames: List[Dict[str, Any]], camera: str = 'CAM_FRONT') -> List[Image.Image]:
        """
        Load frames as a list of PIL images for video processing
        
        Args:
            frames: List of frame information from get_scene_frames
            camera: Camera name
            
        Returns:
            List of PIL images
        """
        video_frames = []
        dataroot = Path(self.nusc.dataroot)
        
        for frame_info in frames:
            if camera in frame_info['cameras']:
                img_path = dataroot / frame_info['cameras'][camera]['filename']
                img = Image.open(img_path).convert('RGB')
                video_frames.append(img)
        
        return video_frames
    
    def get_object_prompts(self, frame_idx: int = 0, video_frames: List[Image.Image] = None) -> Dict[int, List[List[float]]]:
        """
        Get object prompts for SAM3 initialization
        This is a simplified version - in practice, you'd use NuScenes annotations
        or an object detector to get initial points
        
        Args:
            frame_idx: Frame index to get prompts for
            video_frames: List of video frames
            
        Returns:
            Dictionary mapping frame indices to point prompts
        """
        # Simple example: center points for demonstration
        # In production, use NuScenes 3D boxes projected to 2D or detection results
        if video_frames and len(video_frames) > frame_idx:
            h, w = video_frames[frame_idx].size
            
            # Example points (you should replace with actual object locations)
            points = [
                [w * 0.5, h * 0.5],  # Center point
                [w * 0.3, h * 0.6],  # Left object
                [w * 0.7, h * 0.6],  # Right object
            ]
            
            return {frame_idx: points}
        
        return {}
    
    def track_scene(self, scene_name: str, camera: str = 'CAM_FRONT') -> Dict[str, Any]:
        """
        Track objects through a NuScenes scene using SAM3
        
        Args:
            scene_name: Name of the NuScenes scene
            camera: Camera to process
            
        Returns:
            Dictionary containing tracking results
        """
        print(f"Processing scene: {scene_name}, camera: {camera}")
        
        # Get scene frames
        frames = self.get_scene_frames(scene_name, camera)
        if not frames:
            print(f"No frames found for scene {scene_name}")
            return {}
        
        print(f"Found {len(frames)} frames")
        
        # Load video frames
        video_frames = self.load_frames_as_video(frames, camera)
        
        # Get initial object prompts (simplified - use actual detections in practice)
        prompts = self.get_object_prompts(frame_idx=0, video_frames=video_frames)
        
        # Process with SAM3
        inputs = self.processor(
            images=video_frames,
            points=prompts,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate masks
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process masks
        masks = self.processor.post_process_video_segmentation(
            outputs,
            original_sizes=[img.size[::-1] for img in video_frames],
            mask_threshold=self.config.mask_threshold,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
        )
        
        # Save results
        results = self.save_results(scene_name, camera, video_frames, masks, frames)
        
        return results
    
    def save_results(
        self, 
        scene_name: str, 
        camera: str,
        video_frames: List[Image.Image], 
        masks: List[torch.Tensor],
        frame_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Save tracking results and visualizations
        
        Args:
            scene_name: Scene name
            camera: Camera name
            video_frames: Original video frames
            masks: Generated masks
            frame_info: Frame metadata
            
        Returns:
            Dictionary with save paths and statistics
        """
        scene_dir = self.output_dir / scene_name / camera
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'scene_name': scene_name,
            'camera': camera,
            'num_frames': len(video_frames),
            'num_masks': len(masks),
            'output_dir': str(scene_dir),
            'frame_tokens': [f['sample_token'] for f in frame_info]
        }
        
        if self.config.save_masks:
            # Save masks as numpy arrays
            masks_dir = scene_dir / 'masks'
            masks_dir.mkdir(exist_ok=True)
            
            for idx, mask in enumerate(masks):
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                    np.save(masks_dir / f'mask_{idx:04d}.npy', mask_np)
        
        if self.config.save_visualizations:
            # Save visualizations
            vis_dir = scene_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            for idx, (frame, mask) in enumerate(zip(video_frames, masks)):
                # Convert PIL to numpy
                frame_np = np.array(frame)
                
                # Overlay mask
                if isinstance(mask, torch.Tensor):
                    mask_np = mask.cpu().numpy()
                    
                    # Create colored overlay
                    overlay = frame_np.copy()
                    if mask_np.ndim == 3:  # Multiple objects
                        colors = np.random.randint(0, 255, (mask_np.shape[0], 3))
                        for obj_idx in range(mask_np.shape[0]):
                            mask_binary = mask_np[obj_idx] > 0.5
                            overlay[mask_binary] = overlay[mask_binary] * 0.5 + colors[obj_idx] * 0.5
                    else:  # Single mask
                        mask_binary = mask_np > 0.5
                        overlay[mask_binary] = overlay[mask_binary] * 0.5 + np.array([0, 255, 0]) * 0.5
                    
                    # Save visualization
                    cv2.imwrite(
                        str(vis_dir / f'frame_{idx:04d}.jpg'),
                        cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    )
        
        return results


def example_usage():
    """Example usage of NuScenesSAM3Tracker"""
    
    # Create configuration
    config = SAM3DPreprocessorConfig(
        cameras=['CAM_FRONT'],
        max_frames_per_scene=10,  # Process only first 10 frames for demo
        save_masks=True,
        save_visualizations=True,
        output_dir='./sam3_tracking_output'
    )
    
    # Initialize tracker
    tracker = NuScenesSAM3Tracker(config)
    
    # Get first scene name
    first_scene = tracker.nusc.scene[0]['name']
    print(f"Processing scene: {first_scene}")
    
    # Track objects in the scene
    results = tracker.track_scene(first_scene, camera='CAM_FRONT')
    
    print(f"\nTracking completed!")
    print(f"Results saved to: {results['output_dir']}")
    print(f"Processed {results['num_frames']} frames")
    print(f"Generated {results['num_masks']} masks")


if __name__ == "__main__":
    example_usage()
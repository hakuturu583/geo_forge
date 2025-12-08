"""Data classes for SAM3D preprocessing configuration"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SAM3DPreprocessorConfig:
    """Configuration for SAM3D preprocessing on NuScenes dataset"""
    
    # Target object classes for segmentation
    target_classes: List[str] = field(default_factory=lambda: [
        'vehicle.car',
        'vehicle.truck', 
        'vehicle.bus',
        'vehicle.trailer',
        'vehicle.construction',
        'vehicle.emergency',
        'vehicle.motorcycle',
        'vehicle.bicycle',
        'human.pedestrian.adult',
        'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer',
    ])
    
    # Camera configurations
    cameras: List[str] = field(default_factory=lambda: [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT', 
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_FRONT_LEFT'
    ])
    
    # SAM3 model settings
    model_id: str = "facebook/sam2-hiera-large"
    device: str = "cuda"
    
    # Video tracking parameters
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.92
    mask_threshold: float = 0.0
    
    # Processing parameters
    max_frames_per_scene: Optional[int] = None  # Process all frames if None
    batch_size: int = 1
    
    # Output settings
    save_masks: bool = True
    save_visualizations: bool = True
    output_dir: str = "./sam3_outputs"
    
    # NuScenes specific
    version: str = "v1.0-mini"
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'target_classes': self.target_classes,
            'cameras': self.cameras,
            'model_id': self.model_id,
            'device': self.device,
            'points_per_batch': self.points_per_batch,
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
            'mask_threshold': self.mask_threshold,
            'max_frames_per_scene': self.max_frames_per_scene,
            'batch_size': self.batch_size,
            'save_masks': self.save_masks,
            'save_visualizations': self.save_visualizations,
            'output_dir': self.output_dir,
            'version': self.version,
            'verbose': self.verbose,
        }
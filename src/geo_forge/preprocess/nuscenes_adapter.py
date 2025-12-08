"""NuScenes adapter for SAM3 preprocessor"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from PIL import Image
from dotenv import load_dotenv

from nuscenes.nuscenes import NuScenes
from .sam3_preprocessor import VideoDataAdapter
from ..nuscenes import iterate_synchronized_samples

# Load environment variables
load_dotenv()


class NuScenesAdapter(VideoDataAdapter):
    """Adapter for processing NuScenes dataset with SAM3Preprocessor"""

    def __init__(
        self,
        version: str = "v1.0-mini",
        dataroot: Optional[str] = None,
        cameras: Optional[List[str]] = None,
        max_frames: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize NuScenes adapter

        Args:
            version: NuScenes version
            dataroot: Path to NuScenes data (uses env variable if None)
            cameras: List of cameras to use (default all 6)
            max_frames: Maximum frames per scene to process
            verbose: Whether to print progress
        """
        self.version = version
        self.dataroot = dataroot or os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
        self.max_frames = max_frames
        self.verbose = verbose

        # Default cameras
        self.cameras = cameras or [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]

        # Initialize NuScenes
        self.nusc = NuScenes(
            version=self.version, dataroot=self.dataroot, verbose=self.verbose
        )

        # Cache for scene data
        self._scene_cache = {}

    def get_frames(
        self,
        identifier: Union[str, Dict[str, str]],
        camera: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Get video frames for a NuScenes scene

        Args:
            identifier: Scene name (str) or dict with 'scene' and 'camera' keys
            camera: Camera to use (overrides identifier if dict)
            **kwargs: Additional arguments

        Returns:
            List of PIL images
        """
        scene_name, camera_name = self._parse_identifier(identifier, camera)

        # Get cached or load scene data
        cache_key = f"{scene_name}_{camera_name}"
        if cache_key not in self._scene_cache:
            self._scene_cache[cache_key] = self._load_scene_data(
                scene_name, camera_name
            )

        scene_data = self._scene_cache[cache_key]

        # Load frames as images
        frames = []
        dataroot = Path(self.dataroot)

        for frame_info in scene_data["frames"]:
            if camera_name in frame_info["cameras"]:
                img_path = dataroot / frame_info["cameras"][camera_name]["filename"]
                img = Image.open(img_path).convert("RGB")
                frames.append(img)

                if self.max_frames and len(frames) >= self.max_frames:
                    break

        if self.verbose:
            print(
                f"Loaded {len(frames)} frames from scene: {scene_name}, camera: {camera_name}"
            )

        return frames

    def get_prompts(
        self,
        identifier: Union[str, Dict[str, str]],
        frame_idx: int = 0,
        use_annotations: bool = True,
        **kwargs,
    ) -> Dict[int, List[List[float]]]:
        """
        Get object prompts from NuScenes annotations

        Args:
            identifier: Scene identifier
            frame_idx: Frame index for initial prompts
            use_annotations: Whether to use NuScenes 3D box annotations
            **kwargs: Additional arguments

        Returns:
            Dictionary mapping frame indices to point prompts
        """
        scene_name, camera_name = self._parse_identifier(identifier)

        if use_annotations:
            # Get prompts from 3D box projections
            return self._get_annotation_prompts(scene_name, camera_name, frame_idx)
        else:
            # Simple default prompts
            return self._get_default_prompts(frame_idx)

    def get_metadata(
        self, identifier: Union[str, Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        """
        Get metadata for the video sequence

        Args:
            identifier: Scene identifier
            **kwargs: Additional arguments

        Returns:
            Dictionary with metadata
        """
        scene_name, camera_name = self._parse_identifier(identifier)

        # Get scene info
        scene = None
        for s in self.nusc.scene:
            if s["name"] == scene_name:
                scene = s
                break

        if not scene:
            return {"error": f"Scene {scene_name} not found"}

        # Get cached scene data
        cache_key = f"{scene_name}_{camera_name}"
        if cache_key in self._scene_cache:
            scene_data = self._scene_cache[cache_key]
            frame_tokens = [f["sample_token"] for f in scene_data["frames"]]
        else:
            frame_tokens = []

        return {
            "scene_name": scene_name,
            "scene_token": scene["token"],
            "camera": camera_name,
            "description": scene.get("description", ""),
            "num_samples": scene["nbr_samples"],
            "frame_tokens": frame_tokens,
            "dataroot": self.dataroot,
            "version": self.version,
        }

    def _parse_identifier(
        self, identifier: Union[str, Dict[str, str]], camera: Optional[str] = None
    ) -> tuple[str, str]:
        """Parse identifier to get scene name and camera"""
        if isinstance(identifier, str):
            scene_name = identifier
            camera_name = camera or self.cameras[0]
        elif isinstance(identifier, dict):
            scene_name = identifier.get("scene", identifier.get("scene_name"))
            camera_name = camera or identifier.get("camera", self.cameras[0])
        else:
            raise ValueError(f"Invalid identifier type: {type(identifier)}")

        return scene_name, camera_name

    def _load_scene_data(self, scene_name: str, camera_name: str) -> Dict[str, Any]:
        """Load and cache scene data"""
        frames = []
        for sample_info in iterate_synchronized_samples(
            self.nusc, scene_names=[scene_name], cameras=[camera_name]
        ):
            frames.append(sample_info)
            if self.max_frames and len(frames) >= self.max_frames:
                break

        return {"scene_name": scene_name, "camera": camera_name, "frames": frames}

    def _get_annotation_prompts(
        self, scene_name: str, camera_name: str, frame_idx: int
    ) -> Dict[int, List[List[float]]]:
        """Get prompts from NuScenes 3D box annotations"""
        # This is a simplified implementation
        # In practice, you would project 3D boxes to 2D camera coordinates
        # and use those as prompts

        cache_key = f"{scene_name}_{camera_name}"
        if cache_key not in self._scene_cache:
            self._scene_cache[cache_key] = self._load_scene_data(
                scene_name, camera_name
            )

        scene_data = self._scene_cache[cache_key]

        if frame_idx >= len(scene_data["frames"]):
            return {}

        # Get sample token for the frame
        sample_token = scene_data["frames"][frame_idx]["sample_token"]
        sample = self.nusc.get("sample", sample_token)

        # Get annotations for this sample
        points = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Filter by category (vehicles and pedestrians)
            if any(cat in ann["category_name"] for cat in ["vehicle", "human"]):
                # Simplified: use center of 3D box projected to 2D
                # In practice, use nuScenes utilities for proper projection
                # This is just a placeholder
                points.append([320 + len(points) * 50, 240 + len(points) * 30])

                if len(points) >= 5:  # Limit number of prompts
                    break

        if not points:
            # Default points if no annotations
            points = [[320, 240]]

        return {frame_idx: points}

    def _get_default_prompts(self, frame_idx: int) -> Dict[int, List[List[float]]]:
        """Get default prompts when annotations are not used"""
        # Simple grid of points
        points = [
            [320, 240],  # Center
            [200, 300],  # Left
            [440, 300],  # Right
        ]
        return {frame_idx: points}

    def list_scenes(self) -> List[str]:
        """Get list of all scene names"""
        return [s["name"] for s in self.nusc.scene]

    def list_cameras(self) -> List[str]:
        """Get list of available cameras"""
        return self.cameras


def example_usage():
    """Example usage of NuScenesAdapter with SAM3Preprocessor"""
    from .sam3_preprocessor import SAM3Preprocessor
    from ..dataclass import SAM3PreprocessorConfig

    # Initialize adapter
    adapter = NuScenesAdapter(
        version="v1.0-mini", cameras=["CAM_FRONT"], max_frames=10, verbose=True
    )

    # List available scenes
    scenes = adapter.list_scenes()
    print(f"Available scenes: {scenes[:3]}...")

    # Create configuration
    config = SAM3PreprocessorConfig(output_dir="./nuscenes_sam3_output", verbose=True)

    # Initialize SAM3 preprocessor
    preprocessor = SAM3Preprocessor(config=config)

    # Process first scene
    if scenes:
        scene_name = scenes[0]
        print(f"\nProcessing scene: {scene_name}")

        # Process with adapter
        results = preprocessor.process_with_adapter(
            adapter=adapter, identifier={"scene": scene_name, "camera": "CAM_FRONT"}
        )

        print(f"\nResults:")
        print(f"  Scene: {results['metadata']['scene_name']}")
        print(f"  Camera: {results['metadata']['camera']}")
        print(f"  Frames processed: {results['num_frames']}")
        print(f"  Output: {results['output_dir']}")


if __name__ == "__main__":
    example_usage()

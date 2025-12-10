from transformers import Sam3Processor, Sam3Model
import torch
import numpy as np
from PIL import Image
from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from typing import List, Sequence, Dict, Any
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


class SAM3Preprocessor:
    def __init__(self, model_name: str = "facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def generate_attribute_mask(
        self, image: Image.Image, attribute_prompt: str
    ) -> List[ObjectMask]:
        """Generate segmentation masks for the given attribute prompt using SAM3."""
        inputs = self.processor(
            images=image, text=attribute_prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        return ObjectMask.from_result_list(results)

    def generate_masks_from_boxes(
        self,
        nusc: NuScenes,
        camera_data: Dict[str, Any],
        image: Image.Image,
        boxes: Sequence[NuscenesObjectBoundingBox],
    ) -> List[ObjectMask]:
        """
        Generate segmentation masks for the given NuScenes bounding boxes.

        Note: SAM3 does not currently accept box prompts directly, so this
        function projects 3D boxes to the camera plane and fills rectangular
        masks.
        """
        if not boxes:
            return []

        calibrated_sensor = nusc.get(
            "calibrated_sensor", camera_data["calibrated_sensor_token"]
        )
        ego_pose = nusc.get("ego_pose", camera_data["ego_pose_token"])

        cs_trans = np.array(calibrated_sensor["translation"])
        cs_rot = Quaternion(calibrated_sensor["rotation"])
        ego_trans = np.array(ego_pose["translation"])
        ego_rot = Quaternion(ego_pose["rotation"])
        intrinsic = np.array(calibrated_sensor["camera_intrinsic"])

        width, height = image.size
        masks: List[ObjectMask] = []
        for box in boxes:
            # Build NuScenes Box in global frame
            nusc_box = Box(
                center=np.array(box.translation),
                size=np.array(box.size),
                orientation=Quaternion(box.rotation),
            )

            # Transform box to ego frame
            nusc_box.translate(-ego_trans)
            nusc_box.rotate(ego_rot.inverse)

            # Transform box to camera frame
            nusc_box.translate(-cs_trans)
            nusc_box.rotate(cs_rot.inverse)

            # Project corners
            corners = view_points(nusc_box.corners(), intrinsic, normalize=True)
            depths = corners[2, :]
            valid = depths > 0
            if not valid.any():
                continue

            xs = corners[0, valid]
            ys = corners[1, valid]
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            # Clamp to image bounds
            x1_i = int(max(0, min(width - 1, x1)))
            y1_i = int(max(0, min(height - 1, y1)))
            x2_i = int(max(0, min(width - 1, x2)))
            y2_i = int(max(0, min(height - 1, y2)))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            mask = torch.zeros((1, height, width), dtype=torch.bool)
            mask[:, y1_i:y2_i, x1_i:x2_i] = True
            masks.append(ObjectMask(masks=mask))

        return masks

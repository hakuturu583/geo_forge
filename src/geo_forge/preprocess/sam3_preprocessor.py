from transformers import Sam3Processor, Sam3Model
import torch
import numpy as np
from PIL import Image
from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from typing import List, Sequence, Dict, Any
from nuscenes.nuscenes import NuScenes


class SAM3Preprocessor:
    def __init__(self, model_name: str = "facebook/sam3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained(model_name).to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def _run_inference(self, inputs: Any) -> List[ObjectMask]:
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )
        return ObjectMask.from_result_list(results)

    def generate_attribute_mask(
        self, image: Image.Image, attribute_prompt: str
    ) -> List[ObjectMask]:
        """Generate segmentation masks for the given attribute prompt using SAM3."""
        inputs = self.processor(
            images=image, text=attribute_prompt, return_tensors="pt"
        ).to(self.device)
        return self._run_inference(inputs)

    def generate_masks_from_boxes(
        self,
        nusc: NuScenes,
        camera_data: Dict[str, Any],
        image: Image.Image,
        boxes: Sequence[NuscenesObjectBoundingBox],
    ) -> List[ObjectMask]:
        """
        Generate segmentation masks for the given NuScenes bounding boxes.

        Projects 3D boxes into the camera plane and uses them as box prompts.
        """
        width, height = image.size
        sam_condition_boxes: List[List[float]] = []
        for box in boxes:
            bbox_2d = box.to_2d_bbox(
                nusc=nusc,
                calibrated_sensor_token=camera_data["calibrated_sensor_token"],
                ego_pose_token=camera_data["ego_pose_token"],
                image_size=(width, height),
                ignore_category=[
                    "movable_object.barrier",
                    "movable_object.trafficcone",
                    "movable_object.pushable_pullable",
                    "movable_object.debris",
                    "static_object.bicycle_rack",
                ],
            )
            if bbox_2d is None:
                continue

            x_min, y_min, x_max, y_max = bbox_2d
            x0, x1 = int(np.floor(x_min)), int(np.ceil(x_max))
            y0, y1 = int(np.floor(y_min)), int(np.ceil(y_max))
            sam_condition_boxes.append([x0, y0, x1, y1])

        if not sam_condition_boxes:
            return []

        inputs = self.processor(
            images=image,
            input_boxes=[sam_condition_boxes],
            input_boxes_labels=[[1] * len(sam_condition_boxes)],
            return_tensors="pt",
        ).to(self.device)
        return self._run_inference(inputs)

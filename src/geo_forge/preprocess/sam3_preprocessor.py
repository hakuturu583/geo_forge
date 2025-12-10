from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from typing import List, Sequence


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
        self, image: Image.Image, boxes: Sequence[NuscenesObjectBoundingBox]
    ) -> List[ObjectMask]:
        """Generate segmentation masks for the given NuScenes bounding boxes using SAM3."""
        if not boxes:
            return []

        # SAM3 expects [x1, y1, x2, y2] in image pixel coords; NuScenes boxes are 3D.
        # Here we project the 3D boxes into 2D crops via their size and translation,
        # assuming x,y align with image plane for simplicity. Users should replace
        # this with proper projection when available.
        boxes_tensor = torch.tensor(
            [
                [
                    box.translation[0] - box.size[0] / 2,
                    box.translation[1] - box.size[1] / 2,
                    box.translation[0] + box.size[0] / 2,
                    box.translation[1] + box.size[1] / 2,
                ]
                for box in boxes
            ],
            dtype=torch.float32,
        )

        inputs = self.processor(
            images=image, boxes=boxes_tensor, return_tensors="pt"
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

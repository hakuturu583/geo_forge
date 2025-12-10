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
        """
        Generate segmentation masks for the given NuScenes bounding boxes.

        Note: SAM3 does not currently accept box prompts directly, so this
        function returns simple rectangular masks derived from the projected box
        geometry. Supply boxes in image pixel coordinates for best results.
        """
        if not boxes:
            return []

        width, height = image.size
        masks: List[ObjectMask] = []
        for box in boxes:
            x1 = int(box.translation[0] - box.size[0] / 2)
            y1 = int(box.translation[1] - box.size[1] / 2)
            x2 = int(box.translation[0] + box.size[0] / 2)
            y2 = int(box.translation[1] + box.size[1] / 2)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            mask = torch.zeros((1, height, width), dtype=torch.bool)
            mask[:, y1:y2, x1:x2] = True
            masks.append(ObjectMask(masks=mask))

        return masks

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import os
from geo_forge.nuscenes import (
    NuScenes,
    iterate_synchronized_samples,
    load_synchronized_data,
)
from geo_forge.dataclass import ObjectMask
from typing import List


class SAM3MaskPreprocessor:
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


if __name__ == "__main__":
    preprocessor = SAM3MaskPreprocessor()
    dataroot = os.getenv("NUSCENES_DATAROOT", "/data/nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)
    for i, sample_info in enumerate(iterate_synchronized_samples(nusc)):
        print(f"Processing sample {i}")
        pc, images = load_synchronized_data(nusc, sample_info)
        for cam_name, cam_data in images.items():
            image = cam_data["image"]
            attribute_prompt = "sky"
            mask_results = preprocessor.generate_attribute_mask(image, attribute_prompt)
            print(torch.sum(mask_results[0].masks))
        break

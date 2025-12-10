from transformers import Sam3VideoModel, Sam3VideoProcessor
from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from accelerate import Accelerator
import torch
from typing import List


class SAM3VideoPreprocessor:
    def __init__(self, model_name: str = "facebook/sam3"):
        self.device = Accelerator().device
        self.model = Sam3VideoModel.from_pretrained(model_name).to(self.device)
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        self.accelerator = Accelerator()

    def generate_masks_from_video(
        self,
        video_frames: List[torch.Tensor],
        prompts: List[str],
    ):
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )
        self.processor.add_text_prompt(inference_session, prompts)
        outputs_per_frame = {}
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=50
        ):
            processed_outputs = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs
        print(outputs_per_frame)

from transformers import Sam3VideoModel, Sam3VideoProcessor
from geo_forge.dataclass import ObjectMask, NuscenesObjectBoundingBox
from accelerate import Accelerator
import torch
from typing import List


class SAM3VideoPreprocessor:
    def __init__(
        self, model_name: str = "facebook/sam3", dtype: torch.dtype | None = None
    ):
        self.device = Accelerator().device
        if dtype is None:
            # Use bfloat16 on supported CUDA devices, otherwise fall back to float32.
            bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            self.dtype = (
                torch.bfloat16
                if self.device.type == "cuda" and bf16_supported
                else torch.float32
            )
        else:
            self.dtype = dtype

        self.model = Sam3VideoModel.from_pretrained(model_name).to(
            self.device, dtype=self.dtype
        )
        self.processor = Sam3VideoProcessor.from_pretrained(model_name)
        self.accelerator = Accelerator()

    def generate_masks_from_video(
        self,
        video_frames: List[torch.Tensor],
        prompts: List[str],
    ) -> List[List[ObjectMask]]:
        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=self.dtype,
        )
        self.processor.add_text_prompt(inference_session, prompts)
        object_masks: List[List[ObjectMask]] = []
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=50
        ):
            processed_outputs = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            object_masks_per_frame: List[ObjectMask] = []
            object_masks_per_frame.append(
                ObjectMask(
                    masks=processed_outputs["masks"], boxes=processed_outputs["boxes"]
                )
            )

            object_masks.append(object_masks_per_frame)
        return object_masks

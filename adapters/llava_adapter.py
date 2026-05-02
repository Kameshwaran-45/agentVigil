import time
from typing import List, Tuple

import torch
from PIL import Image

from adapters.base import BaseVideoAdapter


class LLaVAAdapter(BaseVideoAdapter):
    """
    LLaVA-OneVision-7B — image-based, 4-bit quantised.
    Original AgentVigil perception model.
    """

    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            LlavaOnevisionForConditionalGeneration,
            BitsAndBytesConfig,
        )
        print(f"[VLM] Loading LLaVA: {self.model_id} ...")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.loaded = True
        print(f"[VLM] LLaVA ready on {next(self.model.parameters()).device}")

    def caption_chunk(
        self,
        frame_paths: List[str],
        prompt_type: str = "standard",
        prev_context: str = "",
    ) -> Tuple[str, float]:
        if not self.loaded:
            self.load()

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        system_prompt, user_text = self._get_prompt(len(images), prompt_type)
        if prev_context:
            user_text = (
                f"Context from previous chunk:\n{prev_context}\n\n"
                f"{user_text}"
            )

        image_content = [{"type": "image"} for _ in images]
        user_content = image_content + [
            {"type": "text", "text": f"{system_prompt}\n\n{user_text}"}
        ]
        messages = [{"role": "user", "content": user_content}]
        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        start = time.time()
        inputs = self.processor(
            text=text_input, images=images, return_tensors="pt"
        ).to(self.model.device, dtype=torch.float16)

        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=250, do_sample=False)
        latency = time.time() - start

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return caption, latency

    def unload(self) -> None:
        if self.loaded:
            del self.model, self.processor
            self.model = self.processor = None
            self.loaded = False
            self._cleanup()
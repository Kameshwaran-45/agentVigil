import re
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from adapters.base import BaseVideoAdapter


class VideoLLaMA3Adapter(BaseVideoAdapter):
    MAX_FRAMES_PER_CHUNK = 16

    @staticmethod
    def _normalize_caption_output(raw: str) -> str:
        text = (raw or "").replace("\r\n", "\n").strip()
        text = text.replace("<|im_end|>", "").strip()
        text = re.sub(r"^<brief factual description>\s*", "", text, flags=re.IGNORECASE)

        event_match = re.search(r"EVENT:\s*([A-Za-z_/\s\-]+)", text, flags=re.IGNORECASE)
        event = "Normal_Videos_event"
        if event_match:
            event = event_match.group(1).strip().splitlines()[0].rstrip(".")

        summary_match = re.search(
            r"(?is)Summary:\s*(.+?)(?:\n\s*EVENT:|$)",
            text,
        )
        summary_text = ""
        if summary_match:
            summary_text = " ".join(summary_match.group(1).strip().split())

        body = re.sub(r"(?is)\n?\s*EVENT:\s*[A-Za-z_/\s\-]+.*$", "", text).strip()
        body = re.sub(r"(?is)\n?\s*Summary:\s*.+$", "", body).strip()

        if re.search(r"(?im)^\s*Detailed\s*:\s*$", body):
            detailed_block = re.split(r"(?im)^\s*Detailed\s*:\s*$", body, maxsplit=1)[-1].strip()
        elif body.lower().startswith("description:"):
            detailed_block = body.split(":", 1)[1].strip()
        else:
            detailed_block = body

        detailed_lines = []
        for ln in detailed_block.splitlines():
            stripped = ln.strip().lstrip("-*").strip()
            if stripped:
                detailed_lines.append(stripped)

        if not detailed_lines and detailed_block:
            split_parts = re.split(r"(?<=[.!?])\s+|\s*,\s*", detailed_block)
            detailed_lines = [p.strip() for p in split_parts if p.strip()]

        if not detailed_lines:
            detailed_lines = ["No clear fine-grained actions returned by model."]

        detailed_lines = detailed_lines[:8]

        if not summary_text:
            summary_text = " ".join(detailed_lines[:2]).strip()
        if not summary_text:
            summary_text = "No clear summary returned by model."

        detailed_text = "\n".join(f"- {line}" for line in detailed_lines)
        return f"Detailed:\n{detailed_text}\nSummary: {summary_text}\nEVENT: {event}"

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"[VLM] Loading VideoLLaMA3: {self.model_id} ...")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": str(self._device)},
        ).eval()

        self._model_dtype = next(self.model.parameters()).dtype

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.loaded = True
        print(f"[VLM] VideoLLaMA3 ready on {self._device} ({self._model_dtype})")

    @classmethod
    def _select_frame_paths(cls, frame_paths: List[str]) -> List[str]:
        if len(frame_paths) <= cls.MAX_FRAMES_PER_CHUNK:
            return frame_paths

        idx = np.linspace(
            0,
            len(frame_paths) - 1,
            cls.MAX_FRAMES_PER_CHUNK,
            dtype=int,
        )
        return [frame_paths[i] for i in idx]

    @staticmethod
    def _load_frame_batch(frame_paths: List[str]) -> List[np.ndarray]:
        """
        Build VideoLLaMA3 frame payload as a list of (C, H, W) uint8 arrays.
        """
        frames_chw: List[np.ndarray] = []
        for fp in frame_paths:
            with Image.open(fp) as img:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            frames_chw.append(arr.transpose(2, 0, 1))

        if not frames_chw:
            raise RuntimeError("No valid frames available for model input")

        return frames_chw

    def caption_chunk(
        self,
        frame_paths: List[str],
        prompt_type: str = "standard",
        prev_context: str = "",
        flashback_prior: str = "",
    ) -> Tuple[str, float]:
        if not self.loaded:
            self.load()

        if not frame_paths:
            return "[ERROR] No frames provided", 0.0

        selected_paths = self._select_frame_paths(frame_paths)
        frames_chw = self._load_frame_batch(selected_paths)

        system_prompt, user_text = self._get_prompt(len(selected_paths), prompt_type)

        # ── FIX: weave the Flashback prior + prev_context into user_text
        # using the shared helper so it lands in the same structured form
        # the analyst-style prompt was written to expect. Before this fix
        # the prior was silently dropped because the adapter's signature
        # didn't have a flashback_prior parameter.
        user_text = self._weave_prior_into_user_text(
            user_text, flashback_prior, prev_context
        )

        try:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": frames_chw,
                            "num_frames": len(frames_chw),
                        },
                        {
                            "type": "text",
                            "text": f"{system_prompt}\n\n{user_text}",
                        },
                    ],
                },
            ]

            inputs = self.processor(
                conversation=conversation,
                add_system_prompt=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # ── FIX: Only move tensors to device; only cast pixel_values ──
            inputs = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._model_dtype)

            start = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=380,
                    min_new_tokens=40,
                    do_sample=False,         
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=4,
                )
            latency = time.time() - start

            input_ids = inputs["input_ids"]
            generated_ids = [
                out[len(inp):] for inp, out in zip(input_ids, output_ids)
            ]
            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            caption = decoded[0].strip() if decoded else ""
            if not caption:
                full = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()
                if full:
                    caption = full
                else:
                    return "[ERROR] Empty model output", latency

            return self._normalize_caption_output(caption), latency

        except Exception as e:
            return f"[ERROR] {str(e)[:260]}", 0.0

    def unload(self) -> None:
        if self.loaded:
            del self.model, self.processor
            self.model = self.processor = None
            self.loaded = False
            self._cleanup()
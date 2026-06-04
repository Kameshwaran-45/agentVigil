import re
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from adapters.base import BaseVideoAdapter


class Qwen25VLAdapter(BaseVideoAdapter):
    """
    Qwen2.5-VL-7B-Instruct adapter.

    Uses native video input through Qwen's vision encoder (not
    image-list mode), so temporal reasoning is end-to-end rather
    than per-frame.
    """

    # Same as VideoLLaMA3 for fair comparison; Qwen supports
    # variable frame counts via dynamic FPS but we fix it here
    # to keep the score-mode comparison apples-to-apples.
    MAX_FRAMES_PER_CHUNK = 16

    # Generation hyperparameters — tuned for analyst-style output
    # with structured six-field response.
    MAX_NEW_TOKENS = 100
    MIN_NEW_TOKENS = 20

    # ────────────────────────────────────────────────────────────
    # Output normalisation — adapted from VideoLLaMA3 adapter
    # ────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize_caption_output(raw: str) -> str:
        """
        Parse Qwen2.5-VL output into the canonical
        'Detailed:\\n...\\nSummary: ...\\nEVENT: ...' format the rest
        of the pipeline expects.

        Qwen tends to produce cleaner structured output than VideoLLaMA3,
        but we still defend against:
        - leading whitespace and chat artifacts
        - the model occasionally producing markdown headers (### Detailed:)
        - compound category labels (Robbery / Armed Robbery)
        """
        text = (raw or "").replace("\r\n", "\n").strip()

        # Strip common chat artifacts
        for token in ["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"]:
            text = text.replace(token, "")
        text = text.strip()

        # Strip markdown header decoration around field labels
        text = re.sub(r"^\s*#+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\*+\s*", "", text, flags=re.MULTILINE)

        # Extract EVENT (use last match — defends against few-shot leakage)
        event = "Normal_Videos_event"
        event_matches = re.findall(
            r"EVENT:\s*([A-Za-z_/\s\-]+?)(?:[.\n]|$)",
            text,
            flags=re.IGNORECASE,
        )
        if event_matches:
            event = event_matches[-1].strip().rstrip(".").splitlines()[0].strip()

        # Extract Summary
        summary_match = re.search(
            r"(?is)Summary:\s*(.+?)(?:\n\s*(?:CONFIRMED|EVIDENCE|SEVERITY|EVENT):|$)",
            text,
        )
        summary_text = ""
        if summary_match:
            summary_text = " ".join(summary_match.group(1).strip().split())

        # Strip the structured-field tail from the body to isolate Detailed
        body = text
        for tail in [
            r"(?is)\n?\s*EVENT:\s*[A-Za-z_/\s\-]+.*$",
            r"(?is)\n?\s*SEVERITY:\s*.+?$",
            r"(?is)\n?\s*EVIDENCE:\s*.+?$",
            r"(?is)\n?\s*CONFIRMED:\s*.+?$",
            r"(?is)\n?\s*Summary:\s*.+?$",
        ]:
            body = re.sub(tail, "", body).strip()

        # Pull the Detailed block
        if re.search(r"(?im)^\s*Detailed\s*:\s*$", body):
            detailed_block = re.split(
                r"(?im)^\s*Detailed\s*:\s*$", body, maxsplit=1
            )[-1].strip()
        else:
            detailed_block = re.sub(
                r"(?i)^\s*detailed:\s*", "", body
            ).strip()

        detailed_lines: List[str] = []
        for ln in detailed_block.splitlines():
            stripped = ln.strip().lstrip("-*•").strip()
            if stripped:
                detailed_lines.append(stripped)

        if not detailed_lines and detailed_block:
            split_parts = re.split(
                r"(?<=[.!?])\s+|\s*,\s*", detailed_block
            )
            detailed_lines = [p.strip() for p in split_parts if p.strip()]

        if not detailed_lines:
            detailed_lines = ["No clear fine-grained actions returned by model."]

        detailed_lines = detailed_lines[:8]

        if not summary_text:
            summary_text = " ".join(detailed_lines[:2]).strip()
        if not summary_text:
            summary_text = "No clear summary returned by model."

        detailed_text = "\n".join(f"- {line}" for line in detailed_lines)
        return (
            f"Detailed:\n{detailed_text}\n"
            f"Summary: {summary_text}\n"
            f"EVENT: {event}"
        )

    # ────────────────────────────────────────────────────────────
    # Model loading — 4-bit quantised for 12GB VRAM
    # ────────────────────────────────────────────────────────────
    def load(self) -> None:
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"[VLM] Loading Qwen2.5-VL: {self.model_id} ...")

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 4-bit NF4 quantisation: ~7 GB after load on the 7B model.
        # Compute dtype bfloat16 matches the model's native training
        # precision and avoids range issues with the vision tower.
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",   # eager fallback if flash_attn missing
        ).eval()

        # Vision encoder needs higher resolution than the LM body
        # for the structured outputs Qwen2.5-VL is famous for.
        # min_pixels / max_pixels control the dynamic resolution
        # behaviour — these defaults work well for surveillance.
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,    # ~200k pixels min
            max_pixels=1024 * 28 * 28,   # ~800k pixels max
        )

        # qwen-vl-utils is the official preprocessing helper.
        # Used here for the video frame stacking; falls back to manual
        # stacking if the package isn't available.
        try:
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
            self._has_qwen_utils = True
        except ImportError:
            print(
                "[VLM] qwen-vl-utils not installed — falling back to manual "
                "frame stacking. Install with: pip install qwen-vl-utils"
            )
            self._has_qwen_utils = False

        self.loaded = True
        first_param = next(self.model.parameters())
        print(
            f"[VLM] Qwen2.5-VL ready on {first_param.device} "
            f"({first_param.dtype}, 4-bit)"
        )

    # ────────────────────────────────────────────────────────────
    # Frame selection — uniform sampling matches other adapters
    # ────────────────────────────────────────────────────────────
    @classmethod
    def _select_frame_paths(cls, frame_paths: List[str]) -> List[str]:
        if len(frame_paths) <= cls.MAX_FRAMES_PER_CHUNK:
            return frame_paths
        idx = np.linspace(
            0, len(frame_paths) - 1, cls.MAX_FRAMES_PER_CHUNK, dtype=int
        )
        return [frame_paths[i] for i in idx]

    # ────────────────────────────────────────────────────────────
    # Main inference path
    # ────────────────────────────────────────────────────────────
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

        # Get the analyst-style prompt
        system_prompt, user_text = self._get_prompt(
            len(selected_paths), prompt_type
        )

        # Weave in retrieved scene priors + previous chunk context.
        # The shared helper on BaseVideoAdapter produces the same
        # block structure across all adapters so we can compare
        # per-VLM contributions without confounding prompt drift.
        user_text = self._weave_prior_into_user_text(
            user_text, flashback_prior, prev_context
        )

        try:
            # Build Qwen2.5-VL message format. Note the "video" content
            # block takes a LIST of file paths (or PIL Images); the
            # processor batches them into a tensor stack.
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": selected_paths,
                            "max_pixels": 360 * 420,
                            "fps": 1.0,   # symbolic; we control sampling above
                        },
                        {
                            "type": "text",
                            "text": user_text,
                        },
                    ],
                },
            ]

            # Apply chat template to get the text portion
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Process vision side
            if self._has_qwen_utils:
                image_inputs, video_inputs = self._process_vision_info(messages)
            else:
                # Manual fallback: stack frames as a list of PIL Images
                image_inputs = None
                video_inputs = [
                    [Image.open(p).convert("RGB") for p in selected_paths]
                ]

            inputs = self.processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._device)

            # Cast pixel/video tensors to the model's compute dtype
            for key in ("pixel_values", "pixel_values_videos"):
                if key in inputs and inputs[key] is not None:
                    inputs[key] = inputs[key].to(torch.bfloat16)

            start = time.time()
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    min_new_tokens=self.MIN_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                    repetition_penalty=1.10,
                    no_repeat_ngram_size=4,
                )
            latency = time.time() - start

            # Trim prompt tokens from each output row (Qwen pattern)
            input_ids = inputs["input_ids"]
            trimmed_ids = [
                out[len(inp):] for inp, out in zip(input_ids, output_ids)
            ]

            decoded = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            caption = decoded[0].strip() if decoded else ""

            if not caption:
                # Last-resort decode of the full output
                full = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                if full:
                    caption = full
                else:
                    return "[ERROR] Empty model output", latency

            return self._normalize_caption_output(caption), latency

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return f"[ERROR] OOM: {str(e)[:200]}", 0.0
        except Exception as e:
            return f"[ERROR] {type(e).__name__}: {str(e)[:260]}", 0.0

    # ────────────────────────────────────────────────────────────
    def unload(self) -> None:
        if self.loaded:
            del self.model, self.processor
            self.model = None
            self.processor = None
            self.loaded = False
            self._cleanup()

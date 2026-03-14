"""
Perception Engine — Stage 1: Knowledge Base Construction
=========================================================
WHAT:  Loads LLaVA-OneVision-7B and generates surveillance captions
       from extracted frame images.

WHY:   This is the "eyes" of AgentVigil. Raw frames are meaningless
       to the reasoning system — the VLM converts visual information
       into structured text that the agent can reason about.

HOW:   Frames → surveillance-tuned prompt with few-shot examples
       → LLaVA-OV generates caption + event classification
       → 4-bit quantization keeps VRAM at ~5GB

SELECTED MODEL: LLaVA-OneVision-7B [IMAGE mode]
  Won ALL 4 crime categories in our benchmark:
  - Robbery ✅  Shoplifting ✅  Road Accident ✅  Vandalism ✅
  - Keyword Hit Rate: 1.53 (best of 9 configurations)
  - Latency: ~11.6s per chunk

CONNECTS TO: video_processor.py provides frame_paths
             database.py stores the generated captions
             agent.py reads captions for reasoning
"""

import time
import torch
from typing import List, Tuple
from PIL import Image
from config import PRIMARY_VLM, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, EVENT_CATEGORIES


class PerceptionEngine:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self):
        """Load VLM into GPU. Call once at startup."""
        from transformers import (
            AutoProcessor,
            LlavaOnevisionForConditionalGeneration,
            BitsAndBytesConfig,
        )

        print(f"[VLM] Loading {PRIMARY_VLM}...")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            PRIMARY_VLM,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            PRIMARY_VLM, trust_remote_code=True
        )
        self.loaded = True
        print(f"[VLM] Ready on {next(self.model.parameters()).device}")

    def caption_chunk(self, frame_paths: List[str]) -> Tuple[str, float]:
        """
        Generate surveillance caption from frame images.
        Returns: (caption_text, latency_seconds)
        """
        if not self.loaded:
            self.load()

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        n = len(images)

        user_prompt = (
            f"These are {n} sequential frames from surveillance footage.\n\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"Analyze the temporal progression across all {n} frames. "
            f"Describe the event and classify it."
        )

        image_content = [{"type": "image"} for _ in images]
        user_content = image_content + [
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
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
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=250, do_sample=False
            )
        latency = time.time() - start

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return caption, latency

    def extract_event_type(self, caption: str) -> str:
        """Parse event classification from caption text."""
        cl = caption.lower()

        # Direct match first
        for cat in EVENT_CATEGORIES:
            if cat.lower() in cl:
                return cat

        # Keyword fallback
        keyword_map = {
            "Road Accident / Vehicle Collision": ["accident", "collision", "crash", "rear-end", "wreck"],
            "Robbery / Armed Robbery": ["robbery", "rob", "mugging", "holdup", "snatch"],
            "Fighting / Assault": ["fight", "assault", "punch", "kick", "brawl", "attack"],
            "Shoplifting / Stealing": ["shoplifting", "steal", "theft", "shoplift", "conceal"],
            "Vandalism / Property Damage": ["vandal", "smash", "destroy", "graffiti", "damage"],
            "Arson / Fire": ["arson", "fire", "flames", "burning", "smoke"],
        }

        for event_type, keywords in keyword_map.items():
            if any(kw in cl for kw in keywords):
                return event_type

        return "Normal Activity"

    def unload(self):
        if self.model:
            del self.model, self.processor
            self.model = self.processor = None
            self.loaded = False
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
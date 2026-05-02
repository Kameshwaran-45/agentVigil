import gc
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from prompt_loader import load_prompt, get_default_stem


class BaseVideoAdapter(ABC):
    """
    Contract that every VLM adapter must satisfy.
    All adapters share the same caption_chunk(frame_paths, prompt_type)
    signature so the rest of the pipeline never needs to change.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights into GPU. Called once at startup."""

    @abstractmethod
    def caption_chunk(
        self,
        frame_paths: List[str],
        prompt_type: str = "standard",
        prev_context: str = "",
    ) -> Tuple[str, float]:
        """
        Generate a surveillance caption from a list of JPEG frame paths.

        Args:
            frame_paths:  Paths to extracted frame JPEGs.
            prompt_type:  "standard" | "benchmark" (keys of prompt files)
            prev_context: Description from previous chunk for temporal continuity.

        Returns:
            (caption_text, latency_seconds)
        """

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory."""

    def _cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _get_prompt(n_frames: int, prompt_stem: str) -> Tuple[str, str]:
        """
        Load the prompt variant by file stem (e.g. "standard", "benchmark").
        Falls back to the default stem if the requested one doesn't exist.
        Returns (system_prompt, user_text).
        """
        try:
            variant = load_prompt(prompt_stem)
        except FileNotFoundError:
            variant = load_prompt(get_default_stem())
        system_prompt = variant["system"]
        few_shot = variant["few_shot"]
        user_text = (
            f"Analyze this sequential batch of {n_frames} surveillance frames.\n\n"
            f"{few_shot}\n\n"
            "Focus on temporal progression and fine-grained actions across the frame sequence. "
            "Produce 4-8 detailed bullet lines, then a 1-2 line summary, then EVENT."
        )
        return system_prompt, user_text
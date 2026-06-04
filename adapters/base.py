import gc
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from prompt_loader import load_prompt, get_default_stem


class BaseVideoAdapter(ABC):
    """
    Contract that every VLM adapter must satisfy.

    UPDATED for hybrid Flashback mode:
    caption_chunk now accepts a flashback_prior block that the
    PerceptionEngine forwards from the gate. Adapters MUST accept
    it (default "") and SHOULD weave it into the prompt so the VLM
    can verify the gate's retrieved scene hypotheses against the
    actual frames.

    See FLASHBACK_FEED_CAPTIONS_TO_VLM and FLASHBACK_VLM_PRIOR_K
    in config.py.
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
        flashback_prior: str = "",
    ) -> Tuple[str, float]:
        """
        Generate a surveillance caption from a list of JPEG frame paths.

        Args:
            frame_paths:     Paths to extracted frame JPEGs.
            prompt_type:     "standard" | "benchmark" (keys of prompt files).
            prev_context:    Description from previous chunk for temporal continuity.
            flashback_prior: Pre-formatted text block with the top-K scene
                             captions retrieved by the Flashback gate. The VLM
                             should treat these as hypotheses to verify, not
                             ground truth to parrot. Empty string when the
                             gate didn't fire or hybrid mode is disabled.

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

    # ── Prior weaving helper (shared by all adapters) ──────────────
    @staticmethod
    def _weave_prior_into_user_text(
        user_text: str,
        flashback_prior: str,
        prev_context: str = "",
    ) -> str:
        """
        Standard formatting block for the gate's retrieved scene captions.

        Adapters call this once per chunk to prepend prior + prev_context
        to user_text in a consistent format the analyst-style prompt
        expects.
        """
        blocks = []

        if flashback_prior:
            blocks.append(
                "### Gate hypotheses (top-K retrieved scenes — VERIFY against frames, "
                "do NOT copy):\n"
                f"{flashback_prior.strip()}\n\n"
                "Treat these as candidates the retrieval gate thinks the video "
                "resembles. The gate is often right but sometimes wrong: a similar-"
                "looking scene can be behaviorally different. Confirm only what "
                "the frames actually show. If the gate's category is wrong, say so "
                "in EVIDENCE and tag the correct EVENT."
            )

        if prev_context:
            blocks.append(
                "### Previous chunk summary (temporal continuity only — do NOT copy):\n"
                f"{prev_context.strip()}"
            )

        blocks.append(
            "### Current chunk — analyze the frames below:\n"
            f"{user_text}"
        )

        return "\n\n---\n\n".join(blocks)
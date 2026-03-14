"""
CLIP Pre-Filter — Between Stage 0 (Chunking) and Stage 1 (VLM)
=================================================================
WHAT:  Scores every chunk for anomaly likelihood using CLIP zero-shot.
       Only chunks above threshold proceed to the expensive VLM.

WHY:   80-90% of surveillance chunks are normal. CLIP scores a chunk
       in ~0.1s vs VLM at ~11.6s. Filters out the boring stuff.

INSTALL: pip install git+https://github.com/openai/CLIP.git ftfy regex
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple

from config import (
    CLIP_ENABLED, CLIP_MODEL, CLIP_ANOMALY_THRESHOLD,
    CLIP_ANOMALY_PROMPTS, CLIP_NORMAL_PROMPTS,
)


class CLIPFilter:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.anomaly_features = None
        self.normal_features = None
        self.device = None
        self.loaded = False

    def load(self):
        if not CLIP_ENABLED:
            print("[CLIP] Disabled in config. All chunks will pass through.")
            return

        try:
            import clip
        except ImportError:
            print("[CLIP] ⚠️ CLIP not installed. Disabling filter.")
            print("[CLIP] Install with: pip install git+https://github.com/openai/CLIP.git ftfy regex")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Loading {CLIP_MODEL} on {self.device}...")

        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)

        with torch.no_grad():
            anomaly_tokens = clip.tokenize(CLIP_ANOMALY_PROMPTS).to(self.device)
            normal_tokens = clip.tokenize(CLIP_NORMAL_PROMPTS).to(self.device)

            self.anomaly_features = self.model.encode_text(anomaly_tokens)
            self.anomaly_features /= self.anomaly_features.norm(dim=-1, keepdim=True)

            self.normal_features = self.model.encode_text(normal_tokens)
            self.normal_features /= self.normal_features.norm(dim=-1, keepdim=True)

        self.loaded = True
        print(f"[CLIP] Ready | {len(CLIP_ANOMALY_PROMPTS)} anomaly prompts, "
              f"{len(CLIP_NORMAL_PROMPTS)} normal prompts")

    def score_frame(self, image: Image.Image) -> float:
        if not self.loaded:
            return 0.5

        img_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            anomaly_sim = (img_features @ self.anomaly_features.T).mean().item()
            normal_sim = (img_features @ self.normal_features.T).mean().item()

        score = (anomaly_sim - normal_sim + 1) / 2
        return max(0.0, min(1.0, score))

    def score_chunk(self, frame_paths: List[str]) -> Tuple[float, str]:
        if not CLIP_ENABLED or not self.loaded:
            return 0.5, "disabled"

        if len(frame_paths) == 0:
            return 0.0, "empty"

        scores = []

        if len(frame_paths) == 1:
            img = Image.open(frame_paths[0]).convert("RGB")
            scores.append(self.score_frame(img))
        elif len(frame_paths) == 2:
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                scores.append(self.score_frame(img))
        else:
            indices = [0, len(frame_paths) // 2, len(frame_paths) - 1]
            for idx in indices:
                img = Image.open(frame_paths[idx]).convert("RGB")
                scores.append(self.score_frame(img))

        max_score = max(scores)
        return round(max_score, 4), "max_of_sampled"

    def filter_chunks(
        self,
        chunks: Dict[int, Dict[str, Any]],
        threshold: float = None,
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[str, Any]]:
        """
        Split chunks into passed (→ VLM) and skipped (→ auto Normal).

        Returns:
            passed_chunks, skipped_chunks, stats_dict
        """
        threshold = threshold or CLIP_ANOMALY_THRESHOLD

        if not CLIP_ENABLED or not self.loaded:
            return chunks, {}, {
                "enabled": False,
                "total": len(chunks),
                "passed": len(chunks),
                "skipped": 0,
                "filter_rate": 0.0,
                "compute_saved_pct": 0.0,
                "scores": {},
            }

        passed = {}
        skipped = {}

        for cidx, cinfo in chunks.items():
            score, strategy = self.score_chunk(cinfo["frame_paths"])
            cinfo["clip_score"] = score
            cinfo["clip_strategy"] = strategy

            if score >= threshold:
                passed[cidx] = cinfo
            else:
                skipped[cidx] = cinfo

        total = len(chunks)
        n_skipped = len(skipped)
        filter_rate = n_skipped / total if total > 0 else 0

        stats = {
            "enabled": True,
            "threshold": threshold,
            "total": total,
            "passed": len(passed),
            "skipped": n_skipped,
            "filter_rate": round(filter_rate, 3),
            "compute_saved_pct": round(filter_rate * 100, 1),
            "scores": {
                cidx: cinfo.get("clip_score", 0)
                for cidx, cinfo in {**passed, **skipped}.items()
            },
        }

        print(f"[CLIP] {len(passed)}/{total} chunks passed "
              f"(threshold={threshold}) — "
              f"{stats['compute_saved_pct']}% filtered out")

        return passed, skipped, stats

    def unload(self):
        if self.model:
            del self.model, self.preprocess
            del self.anomaly_features, self.normal_features
            self.model = None
            self.loaded = False
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
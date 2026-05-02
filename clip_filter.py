"""
CLIP Pre-Filter — Between Stage 0 (Chunking) and Stage 1 (VLM)
=================================================================
WHAT:  Scores every chunk for anomaly likelihood using CLIP zero-shot.
       Only chunks above threshold proceed to the expensive VLM.

WHY:   80-90% of surveillance chunks are normal. CLIP scores a chunk
       in ~0.1s vs VLM at ~11.6s. Filters out the boring stuff.

PROMPT SOURCE
-------------
Anomaly and normal prompts are now loaded from the PostgreSQL
clip_prompts table (seeded by scripts/upload_clip_prompts.py).
Adding more prompts improves discrimination WITHOUT code changes.

  Scoring formula:
      score = (mean_anomaly_sim - mean_normal_sim + 1) / 2

  More anomaly prompts → real crime frames hit more prompts → mean_anomaly_sim rises
  More normal prompts  → real normal frames score high on normal side → lower score
  Wider separation → threshold can be tighter → more VLM compute saved

PROMPT MANAGEMENT
-----------------
  Add/edit prompts via SQL or scripts/upload_clip_prompts.py.
  Apply without restart:
      clip_filter.reload_prompts()   # re-fetches DB + re-encodes CLIP features

INSTALL
-------
    pip install git+https://github.com/openai/CLIP.git ftfy regex
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple

from config import CLIP_ENABLED, CLIP_MODEL, CLIP_ANOMALY_THRESHOLD
from clip_prompt_store import CLIPPromptStore


class CLIPFilter:
    def __init__(self):
        self.model       = None
        self.preprocess  = None
        self.device      = None
        self.loaded      = False

        # Text feature matrices (encoded from prompts)
        self.anomaly_features = None
        self.normal_features  = None

        # Prompt store — set during load() when pg_conn is available
        self._prompt_store: Optional[CLIPPromptStore] = None

    # ── LOAD ────────────────────────────────────────────────────────

    def load(self, pg_conn=None) -> None:
        """
        Load CLIP model and encode prompts from DB.

        Args:
            pg_conn: psycopg2 connection from DatabaseManager.
                     Pass this so prompts are loaded from DB.
                     If None, falls back to config.py defaults.
        """
        if not CLIP_ENABLED:
            print("[CLIP] Disabled in config. All chunks will pass through.")
            return

        try:
            import clip
        except ImportError:
            print("[CLIP] ⚠️ CLIP not installed. Disabling filter.")
            print("[CLIP] Install: pip install git+https://github.com/openai/CLIP.git ftfy regex")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[CLIP] Loading {CLIP_MODEL} on {self.device}...")
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)
        self._clip_module = clip   # keep reference for reload

        # Load prompts from DB (or fallback to config.py)
        if pg_conn is not None:
            self._prompt_store = CLIPPromptStore(pg_conn)
        else:
            # No DB connection yet — use config.py fallback via store
            print("[CLIP] No pg_conn provided — using config.py prompt fallback.")
            self._prompt_store = CLIPPromptStore.__new__(CLIPPromptStore)
            self._prompt_store._conn  = None
            self._prompt_store._rows  = []
            from config import CLIP_ANOMALY_PROMPTS, CLIP_NORMAL_PROMPTS
            self._prompt_store.anomaly_texts = list(CLIP_ANOMALY_PROMPTS)
            self._prompt_store.normal_texts  = list(CLIP_NORMAL_PROMPTS)

        self._encode_prompts()
        self.loaded = True

    def _encode_prompts(self) -> None:
        """
        Encode current prompt lists into CLIP text feature matrices.
        Called on load and on reload_prompts().
        The matrix shape is (N_prompts, embedding_dim).
        """
        clip    = self._clip_module
        anomaly = self._prompt_store.anomaly_texts
        normal  = self._prompt_store.normal_texts

        if not anomaly or not normal:
            print("[CLIP] WARNING: empty prompt list — filter will pass all chunks.")
            self.anomaly_features = None
            self.normal_features  = None
            return

        with torch.no_grad():
            a_tokens = clip.tokenize(anomaly).to(self.device)
            n_tokens = clip.tokenize(normal).to(self.device)

            self.anomaly_features = self.model.encode_text(a_tokens)
            self.anomaly_features /= self.anomaly_features.norm(dim=-1, keepdim=True)

            self.normal_features = self.model.encode_text(n_tokens)
            self.normal_features /= self.normal_features.norm(dim=-1, keepdim=True)

        stats = self._prompt_store.stats()
        print(f"[CLIP] Prompts encoded — "
              f"{stats['anomaly_count']} anomaly (source: {stats['source']}), "
              f"{stats['normal_count']} normal")

    # ── RUNTIME PROMPT RELOAD ────────────────────────────────────────

    def reload_prompts(self) -> Dict:
        """
        Re-fetch prompts from DB and re-encode CLIP text features.
        Call after scripts/upload_clip_prompts.py adds new prompts.
        Safe to call mid-session — uses old features until complete.

        Returns stats dict from CLIPPromptStore.
        """
        if not self.loaded:
            print("[CLIP] Not loaded yet — call load() first.")
            return {}
        if self._prompt_store is None:
            print("[CLIP] No prompt store — cannot reload.")
            return {}

        self._prompt_store.reload()
        self._encode_prompts()
        return self._prompt_store.stats()

    # ── SCORING ──────────────────────────────────────────────────────

    def score_frame(self, image: Image.Image) -> float:
        """Score a single PIL image for anomaly likelihood [0.0, 1.0]."""
        if not self.loaded or self.anomaly_features is None:
            return 0.5

        img_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(img_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # Mean cosine similarity across all prompts in each set
            anomaly_sim = (img_features @ self.anomaly_features.T).mean().item()
            normal_sim  = (img_features @ self.normal_features.T).mean().item()

        # Normalise difference to [0, 1]
        score = (anomaly_sim - normal_sim + 1) / 2
        return max(0.0, min(1.0, score))

    def score_chunk(self, frame_paths: List[str]) -> Tuple[float, str]:
        """
        Score a chunk by sampling up to 3 frames and taking the max.
        Returns (score, strategy_label).
        """
        if not CLIP_ENABLED or not self.loaded:
            return 0.5, "disabled"
        if not frame_paths:
            return 0.0, "empty"

        n = len(frame_paths)
        if n == 1:
            indices, strategy = [0], "single"
        elif n == 2:
            indices, strategy = [0, 1], "both"
        else:
            indices = [0, n // 2, n - 1]
            strategy = "max_of_sampled"

        scores = []
        for idx in indices:
            img = Image.open(frame_paths[idx]).convert("RGB")
            scores.append(self.score_frame(img))

        return round(max(scores), 4), strategy

    # ── FILTER ───────────────────────────────────────────────────────

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
        threshold = threshold if threshold is not None else CLIP_ANOMALY_THRESHOLD

        if not CLIP_ENABLED or not self.loaded:
            return chunks, {}, {
                "enabled": False,
                "total": len(chunks),
                "passed": len(chunks),
                "skipped": 0,
                "filter_rate": 0.0,
                "compute_saved_pct": 0.0,
                "scores": {},
                "prompt_stats": {},
            }

        passed  = {}
        skipped = {}

        for cidx, cinfo in chunks.items():
            score, strategy = self.score_chunk(cinfo["frame_paths"])
            cinfo["clip_score"]    = score
            cinfo["clip_strategy"] = strategy

            if score >= threshold:
                passed[cidx]  = cinfo
            else:
                skipped[cidx] = cinfo

        total       = len(chunks)
        n_skipped   = len(skipped)
        filter_rate = n_skipped / total if total > 0 else 0

        prompt_stats = self._prompt_store.stats() if self._prompt_store else {}

        stats = {
            "enabled":           True,
            "threshold":         threshold,
            "total":             total,
            "passed":            len(passed),
            "skipped":           n_skipped,
            "filter_rate":       round(filter_rate, 3),
            "compute_saved_pct": round(filter_rate * 100, 1),
            "scores": {
                cidx: cinfo.get("clip_score", 0)
                for cidx, cinfo in {**passed, **skipped}.items()
            },
            "prompt_stats": prompt_stats,
        }

        print(f"[CLIP] {len(passed)}/{total} passed "
              f"(threshold={threshold}, "
              f"{prompt_stats.get('anomaly_count', '?')}a/"
              f"{prompt_stats.get('normal_count', '?')}n prompts) — "
              f"{stats['compute_saved_pct']}% filtered out")

        return passed, skipped, stats

    # ── UNLOAD ───────────────────────────────────────────────────────

    def unload(self) -> None:
        if self.model:
            del self.model, self.preprocess
            del self.anomaly_features, self.normal_features
            self.model           = None
            self.anomaly_features = None
            self.normal_features  = None
            self.loaded          = False
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── INFO ─────────────────────────────────────────────────────────

    def prompt_stats(self) -> Dict:
        """Return current prompt count stats for UI display."""
        if self._prompt_store:
            return self._prompt_store.stats()
        return {"anomaly_count": 0, "normal_count": 0, "source": "not_loaded"}
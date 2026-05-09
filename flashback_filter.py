"""
Flashback Filter — Phase 2 replacement for CLIPFilter
======================================================
WHAT:  The online stage of Flashback. Encodes a video chunk with PE,
       retrieves the top-K most similar pseudo-scene captions from
       memory, computes a softmax-weighted anomaly score, and returns
       both the score AND the retrieved captions for downstream use.

ROLE IN PHASE-2 PIPELINE (HYBRID MODE)
---------------------------------------
This filter does TWO jobs the old CLIPFilter could not:

    1. GATE — chunks below CLIP_ANOMALY_THRESHOLD skip the VLM and
       are stored directly as Normal. Same role as CLIPFilter, but
       calibrated against millions of captions instead of ~12.

    2. PRIOR — for chunks that pass, the retrieved captions and
       categories are passed to the VLM as a prior context block.
       The VLM still produces the canonical Detailed/Summary/EVENT
       caption, but starts from PE's view of the scene.

       This is the "hybrid" choice from the user's design call:
       PE replaces the gate AND seeds richer captioning.

SCORE FORMULA  (paper §3.3)
----------------------------
    Given segment feature v (L2-normalised, 1024-D PE features):

        σ_j = v · t_j           # dot product against every memory entry j
        J   = top-K indices of σ
        w_k = softmax(σ_J)_k    # K weights summing to 1
        A   = Σ_k w_k · y_{J_k} # anomaly flag of each retrieved entry

    A ∈ [0, 1] — direct probability that the chunk is anomalous.
    The retrieved captions c_{J_k} are returned as explanations.

WHY THIS BEATS CLIP
-------------------
CLIPFilter scored a chunk by comparing it to ~12 anomaly prompts and
~7 normal prompts, then taking (anomaly_sim − normal_sim + 1) / 2.
With 19 prompts total, real anomalies that didn't phonetically match
the prompts got dropped. Flashback compares against potentially 1M
captions covering every scene the LLM could imagine, so coverage is
no longer the bottleneck.

SAP IS ALREADY APPLIED — the anomalous embeddings in memory are
pre-scaled by α=0.95. We do NOT re-apply it here; that would be
double penalisation.

CONNECTS TO:
    perception_encoder.py     — encode_video for the segment
    pseudo_scene_memory.py    — load_all on startup
    app.py                    — calls filter_chunks(...)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from perception_encoder import PerceptionEncoder
from pseudo_scene_memory import PseudoSceneMemory


class FlashbackFilter:
    """
    Drop-in replacement for CLIPFilter with a richer return contract.

    PUBLIC METHODS (called from app.py)
    -------------------------------------
        load(pg_conn)                                  — startup
        filter_chunks(chunks, threshold=...)           — bulk gate
        score_chunk(frame_paths)                       — single chunk
        prompt_stats()                                 — UI display

    The bulk filter writes per-chunk metadata back into the chunks dict:
        chunk["flashback_score"]              float
        chunk["flashback_label"]              "ALERT" | "NORMAL"
        chunk["flashback_top_captions"]       List[str]    # for VLM prior
        chunk["flashback_top_categories"]     List[str]
        chunk["flashback_embedding"]          np.ndarray   # reused by Tier 2
    """

    def __init__(
        self,
        top_k: int = 10,                   # paper's K = 10
        encoder_model: str = "PE-Core-G14-448",
    ):
        self.top_k = top_k
        self.encoder_model_name = encoder_model

        self.encoder: Optional[PerceptionEncoder] = None
        self.memory: Optional[PseudoSceneMemory] = None

        # In-memory caption bank — loaded once at startup
        self._cap_texts:    List[str]   = []
        self._cap_cats:     List[str]   = []
        self._cap_embeds:   np.ndarray  = np.zeros((0, 0), dtype=np.float32)
        self._cap_labels:   np.ndarray  = np.zeros((0,),  dtype=np.int8)

        self.loaded = False

    # ── LOAD ────────────────────────────────────────────────────────

    def load(self, pg_conn) -> None:
        """
        Initialise PE + read the entire pseudo-scene memory into RAM.

        We hold the full memory in process — at 1M × 1024 floats this is
        ~4 GiB, which fits on every box that runs PE itself. Replace
        with a Milvus-backed loader if you need >10M entries.
        """
        if self.loaded:
            return

        self.encoder = PerceptionEncoder(model_name=self.encoder_model_name)
        self.encoder.load()

        self.memory = PseudoSceneMemory(pg_conn)

        if self.memory.is_empty():
            print("[Flashback] WARNING: pseudo_scene_memory is empty. "
                  "Run scripts/build_pseudo_scene_memory.py before "
                  "processing videos. Filter will pass everything through.")
            self.loaded = True   # we still mark loaded — degraded mode
            return

        captions, embeds, labels, cats = self.memory.load_all()

        if embeds.shape[1] != self.encoder.embed_dim:
            raise RuntimeError(
                f"Memory embed_dim ({embeds.shape[1]}) does not match "
                f"current encoder ({self.encoder.embed_dim}). "
                "Rebuild the memory with the same PE backbone."
            )

        self._cap_texts  = captions
        self._cap_cats   = cats
        self._cap_embeds = embeds
        self._cap_labels = labels

        self.loaded = True
        print(f"[Flashback] Memory ready — "
              f"{(labels == 0).sum()} normal, {(labels == 1).sum()} anomalous, "
              f"{len(set(cats))} categories, dim={embeds.shape[1]}")

    # ── SINGLE CHUNK ────────────────────────────────────────────────

    def score_chunk(self, frame_paths: List[str]) -> Dict[str, Any]:
        """
        Encode one chunk and run Flashback retrieval.
        Returns a dict with score, top captions, top categories, and the
        L2-normalised PE embedding (reusable by the agent's Tier 2).
        """
        if not self.loaded or self._cap_embeds.size == 0:
            return {
                "score":        0.5,           # neutral
                "top_captions": [],
                "top_categories": [],
                "embedding":    np.zeros((0,), dtype=np.float32),
                "strategy":     "memory_empty",
            }

        v = self.encoder.encode_video(frame_paths)        # (D,) unit vec
        # Dot product across the entire memory
        sims = self._cap_embeds @ v                       # (N,)

        K = min(self.top_k, sims.shape[0])
        top_idx = np.argpartition(-sims, K - 1)[:K]
        # Sort the top-K so explanations come out in similarity order
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        top_sims = sims[top_idx]

        # Softmax over the K retrieved similarities (paper §3.3)
        # Numerical stability: subtract max before exp
        z = top_sims - top_sims.max()
        weights = np.exp(z) / np.exp(z).sum()

        anomaly_score = float(np.dot(weights, self._cap_labels[top_idx]))

        return {
            "score":          round(anomaly_score, 4),
            "top_captions":   [self._cap_texts[i] for i in top_idx],
            "top_categories": [self._cap_cats[i]  for i in top_idx],
            "top_labels":     self._cap_labels[top_idx].tolist(),
            "top_similarities": [round(float(s), 4) for s in top_sims],
            "embedding":      v,
            "strategy":       "pe_retrieve_topk",
        }

    # ── BULK FILTER (called by app.py) ──────────────────────────────

    def filter_chunks(
        self,
        chunks: Dict[int, Dict[str, Any]],
        threshold: float = 0.25,
    ) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[str, Any]]:
        """
        Score every chunk, split into passed (→ VLM) and skipped (→ Normal).
        Same return shape as CLIPFilter.filter_chunks for drop-in compat.

        Per-chunk side-effects on chunks dict:
            flashback_score        float
            flashback_top_captions List[str]    (top-K)
            flashback_top_categories List[str]
            flashback_embedding    np.ndarray   (reused by agent)
        """
        if not self.loaded or self._cap_embeds.size == 0:
            # Degraded mode — pass everything through, mark accordingly
            for cidx, cinfo in chunks.items():
                cinfo["flashback_score"] = 0.5
                cinfo["flashback_top_captions"] = []
                cinfo["flashback_top_categories"] = []
                cinfo["flashback_embedding"] = np.zeros((0,), dtype=np.float32)
            return chunks, {}, {
                "enabled":  False,
                "reason":   "memory_empty_or_unloaded",
                "total":    len(chunks),
                "passed":   len(chunks),
                "skipped":  0,
                "filter_rate":       0.0,
                "compute_saved_pct": 0.0,
                "scores":   {},
            }

        passed:  Dict[int, Dict] = {}
        skipped: Dict[int, Dict] = {}
        scores:  Dict[int, float] = {}

        for cidx, cinfo in chunks.items():
            result = self.score_chunk(cinfo["frame_paths"])
            cinfo["flashback_score"]          = result["score"]
            cinfo["flashback_top_captions"]   = result["top_captions"]
            cinfo["flashback_top_categories"] = result["top_categories"]
            cinfo["flashback_embedding"]      = result["embedding"]
            scores[cidx] = result["score"]

            if result["score"] >= threshold:
                passed[cidx]  = cinfo
            else:
                skipped[cidx] = cinfo

        n = len(chunks)
        filter_rate = (len(skipped) / n) if n else 0.0

        memory_stats = self.memory.stats() if self.memory else {}

        stats = {
            "enabled":           True,
            "threshold":         threshold,
            "total":             n,
            "passed":            len(passed),
            "skipped":           len(skipped),
            "filter_rate":       round(filter_rate, 3),
            "compute_saved_pct": round(filter_rate * 100, 1),
            "scores":            scores,
            "memory_stats":      memory_stats,
            "encoder":           self.encoder_model_name,
            "top_k":             self.top_k,
        }

        print(f"[Flashback] {len(passed)}/{n} passed "
              f"(threshold={threshold}, K={self.top_k}, "
              f"{memory_stats.get('normal_count','?')}n / "
              f"{memory_stats.get('anomalous_count','?')}a memory) — "
              f"{stats['compute_saved_pct']}% filtered out")

        return passed, skipped, stats

    # ── INFO ────────────────────────────────────────────────────────

    def prompt_stats(self) -> dict:
        """Compatibility wrapper for the CLIP-era sidebar."""
        if self.memory is None:
            return {"anomaly_count": 0, "normal_count": 0, "source": "not_loaded"}
        s = self.memory.stats()
        return {
            "anomaly_count":   s.get("anomalous_count", 0),
            "normal_count":    s.get("normal_count", 0),
            "category_count":  s.get("category_count", 0),
            "total":           s.get("total", 0),
            "source":          "pseudo_scene_memory",
        }

    def unload(self) -> None:
        if self.encoder is not None:
            self.encoder.unload()
        self.encoder = None
        self.memory  = None
        self._cap_texts = []
        self._cap_cats  = []
        self._cap_embeds = np.zeros((0, 0), dtype=np.float32)
        self._cap_labels = np.zeros((0,),  dtype=np.int8)
        self.loaded = False
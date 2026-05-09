"""
Pseudo-Scene Memory — Flashback's offline knowledge base
==========================================================
WHAT:  Stores LLM-generated normal/anomalous scene captions PLUS their
       Perception Encoder embeddings. At inference time the gate looks
       up the most-similar entries instead of running an LLM.

WHY:   This is the single ingredient that makes Flashback both
       zero-shot and real-time. The LLM does its work ONCE during
       offline construction; online inference is one PE forward pass
       and a batched dot product.

WHERE THE TWO BIG ACCURACY KNOBS LIVE
-------------------------------------
Repulsive Prompting (RP) and Scaled Anomaly Penalisation (SAP) are
applied HERE at memory build time, not during inference. This means:

    1. Every caption we encode for storage is wrapped with RP first.
       RP widens the angle between the normal and anomalous centroids
       in PE feature space (8° → 33° on UCF-Crime in the paper).

    2. After encoding, anomalous embeddings are scaled by α (default
       0.95). SAP attenuates the encoder's structural bias toward
       anomalous text — a smaller-norm anomalous vector loses more
       similarity than a normal one in a dot product.

Because both operations are stored materialised, online retrieval
remains pure dot product. No per-query post-processing needed.

PROMPT TEMPLATES (paper §3.2, "Repulsive prompting")
-----------------------------------------------------
The paper finds the keyword + wrapper combination is what works —
keyword-only or template-only each recovers ~half the gain. We use
both in the canonical form described in the appendix:

    Normal templates:
        "Normal Scene:\\n  Action Category: {category}\\n  Description: {text}"

    Anomalous templates:
        "Anomalous Scene:\\n  Action Category: {category}\\n  Description: {text}"

SCHEMA — pseudo_scene_memory table
-----------------------------------
    id           SERIAL PK
    label        SMALLINT     -- 0 normal, 1 anomalous
    category     TEXT         -- e.g. "Robbery", "Office work"
    raw_caption  TEXT         -- original LLM caption, no RP wrapper
    rp_caption   TEXT         -- final string passed to PE encoder
    embedding    BYTEA        -- float32 PE features, L2-normalised then SAP-scaled
    embed_dim    INTEGER      -- redundant but cheap, sanity check on load
    enabled      BOOLEAN      -- soft-delete / A-B testing
    created_at   TIMESTAMP

INTERFACE
---------
Build-time (offline, called by scripts/build_pseudo_scene_memory.py):
    store.write_pairs([
        {"label": 0, "category": "Office work", "text": "..."},
        {"label": 1, "category": "Robbery",     "text": "..."},
        ...
    ], encoder=pe, alpha=0.95)

Inference-time (called by FlashbackFilter on startup):
    captions, embeddings, labels, categories = store.load_all()
    # embeddings already include RP+SAP — ready for dot product
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


# ── Repulsive Prompting wrappers (paper §3.2) ───────────────────────
# Keep these constants — they are the textual half of RP. Changing them
# requires re-encoding the entire memory.

NORMAL_KEYWORD    = "Normal"
ANOMALOUS_KEYWORD = "Anomalous"

NORMAL_TEMPLATE = (
    "{kw} Scene:\n"
    "  Action Category: {category}\n"
    "  Description: {text}"
)
ANOMALOUS_TEMPLATE = (
    "{kw} Scene:\n"
    "  Action Category: {category}\n"
    "  Description: {text}"
)


def apply_rp(label: int, category: str, text: str) -> str:
    """
    Wrap a caption with Repulsive Prompting.
    label: 0 = normal, 1 = anomalous.
    """
    if label == 0:
        return NORMAL_TEMPLATE.format(
            kw=NORMAL_KEYWORD, category=category.strip(), text=text.strip()
        )
    if label == 1:
        return ANOMALOUS_TEMPLATE.format(
            kw=ANOMALOUS_KEYWORD, category=category.strip(), text=text.strip()
        )
    raise ValueError(f"label must be 0 or 1, got {label}")


@dataclass
class MemoryEntry:
    """One row in the pseudo-scene memory."""
    label: int                  # 0 normal, 1 anomalous
    category: str               # paper's κ — used as explanation tag
    text: str                   # paper's c — raw description from LLM


class PseudoSceneMemory:
    """
    PostgreSQL-backed Flashback memory.

    Embeddings are stored as raw float32 BYTEA — Postgres doesn't need
    vector indexing here because retrieval happens in-process via numpy
    after load_all(). The whole memory (1M × 1024 floats ≈ 4 GiB) is
    read once at startup; that's exactly the trade Flashback prescribes.
    For multi-million-entry deployments, swap to Milvus by adapting
    load_all() — the rest of the code path doesn't change.
    """

    TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS pseudo_scene_memory (
            id           SERIAL PRIMARY KEY,
            label        SMALLINT NOT NULL CHECK (label IN (0,1)),
            category     TEXT NOT NULL,
            raw_caption  TEXT NOT NULL,
            rp_caption   TEXT NOT NULL,
            embedding    BYTEA NOT NULL,
            embed_dim    INTEGER NOT NULL,
            enabled      BOOLEAN NOT NULL DEFAULT TRUE,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (label, rp_caption)
        );
        CREATE INDEX IF NOT EXISTS idx_psm_label    ON pseudo_scene_memory(label);
        CREATE INDEX IF NOT EXISTS idx_psm_category ON pseudo_scene_memory(category);
        CREATE INDEX IF NOT EXISTS idx_psm_enabled  ON pseudo_scene_memory(enabled);
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self.TABLE_DDL)

    # ── BUILD (offline) ─────────────────────────────────────────────

    def write_pairs(
        self,
        entries: Iterable[MemoryEntry],
        encoder,                  # PerceptionEncoder duck-type
        alpha: float = 0.95,
        batch_size: int = 256,
    ) -> int:
        """
        Encode + insert a batch of entries.

        For each entry:
            1. Build RP-wrapped caption.
            2. Encode with PE → 1024-D unit vector.
            3. If anomalous: multiply by alpha (SAP).
            4. Insert (raw, rp, bytes).

        Returns number of rows inserted (skips ON CONFLICT duplicates).
        """
        entries = list(entries)
        if not entries:
            return 0

        rp_texts = [apply_rp(e.label, e.category, e.text) for e in entries]
        labels   = np.array([e.label for e in entries], dtype=np.int8)

        # Encode in batches — encoder handles internal batching too,
        # but we keep the call shapes predictable for the caller.
        feats = encoder.encode_texts(rp_texts, batch_size=batch_size)
        # Apply SAP — anomalous embeddings shrunk by alpha
        anom_mask = (labels == 1)
        if anom_mask.any():
            feats[anom_mask] = feats[anom_mask] * float(alpha)

        # Re-cast to float32 (encoder may return float16 on some HW)
        feats = feats.astype(np.float32, copy=False)
        embed_dim = int(feats.shape[1])

        rows = []
        for entry, rp_text, vec in zip(entries, rp_texts, feats):
            rows.append((
                int(entry.label),
                entry.category,
                entry.text,
                rp_text,
                vec.tobytes(),
                embed_dim,
            ))

        # psycopg2 execute_values for speed at million scale
        from psycopg2.extras import execute_values
        with self._conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO pseudo_scene_memory
                    (label, category, raw_caption, rp_caption, embedding, embed_dim)
                VALUES %s
                ON CONFLICT (label, rp_caption) DO NOTHING
                """,
                rows,
                page_size=500,
            )
            return cur.rowcount

    # ── LOAD (inference startup) ────────────────────────────────────

    def load_all(
        self,
        enabled_only: bool = True,
        max_entries: Optional[int] = None,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[str]]:
        """
        Load the full memory into memory as numpy arrays.

        Returns:
            raw_captions: list of length N            (used for explanations)
            embeddings:   np.ndarray (N, D) float32   (RP+SAP applied)
            labels:       np.ndarray (N,) int8        (0 normal, 1 anomalous)
            categories:   list of length N            (used for explanations)
        """
        with self._conn.cursor() as cur:
            sql = """
                SELECT raw_caption, embedding, embed_dim, label, category
                FROM   pseudo_scene_memory
            """
            if enabled_only:
                sql += " WHERE enabled = TRUE"
            sql += " ORDER BY id"
            if max_entries:
                sql += f" LIMIT {int(max_entries)}"
            cur.execute(sql)
            rows = cur.fetchall()

        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32), \
                   np.zeros((0,), dtype=np.int8), []

        embed_dim = int(rows[0][2])
        n = len(rows)

        captions   = [r[0] for r in rows]
        embeddings = np.empty((n, embed_dim), dtype=np.float32)
        labels     = np.empty(n, dtype=np.int8)
        categories = [r[4] for r in rows]

        for i, (_, blob, dim, lbl, _) in enumerate(rows):
            if dim != embed_dim:
                raise RuntimeError(
                    f"Mixed embed_dim in pseudo_scene_memory: "
                    f"{embed_dim} vs {dim} at row {i}. "
                    "Did you change the encoder mid-build?"
                )
            embeddings[i] = np.frombuffer(blob, dtype=np.float32)
            labels[i]     = lbl

        return captions, embeddings, labels, categories

    # ── STATS ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._conn.cursor() as cur:
            cur.execute("""
                SELECT label, COUNT(*) FROM pseudo_scene_memory
                WHERE enabled = TRUE GROUP BY label
            """)
            counts = {int(lbl): int(n) for lbl, n in cur.fetchall()}
            cur.execute("""
                SELECT COUNT(DISTINCT category) FROM pseudo_scene_memory
                WHERE enabled = TRUE
            """)
            cat_count = int(cur.fetchone()[0])

        return {
            "normal_count":    counts.get(0, 0),
            "anomalous_count": counts.get(1, 0),
            "total":           sum(counts.values()),
            "category_count":  cat_count,
        }

    def is_empty(self) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pseudo_scene_memory LIMIT 1")
            return cur.fetchone() is None
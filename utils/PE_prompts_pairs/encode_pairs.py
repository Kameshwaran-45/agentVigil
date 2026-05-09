"""
encode_pairs.py
=================
Stage 2 of the Flashback memory build.

Reads raw_scene_pairs rows where encoded=FALSE, applies Repulsive
Prompting + Scaled Anomaly Penalisation, runs them through the frozen
PerceptionEncoder, and writes the result into pseudo_scene_memory.

WHY POSTGRES, NOT MILVUS
------------------------
The pseudo-scene memory is a STATIC reference bank — written once,
read once at app startup into a numpy array, never queried by the DB
at runtime. The Flashback paper itself stores this in process RAM and
runs "a million dot products" per chunk via numpy. Milvus is built
for growing, frequently-queried collections — wrong access pattern
for this artifact. Postgres handles the durable storage; the runtime
filter loads everything into RAM on startup.

Milvus stays focused on its real job: the per-chunk caption index for
Tier 2 of the agent, which IS dynamic and IS query-heavy.

REQUIREMENTS
------------
    pip install python-dotenv psycopg2-binary numpy torch pillow
    pip install git+https://github.com/facebookresearch/perception_models

SETUP
-----
.env must contain Postgres credentials. PE downloads its own weights
on first run (~5 GB, takes a few minutes).

USAGE
-----
    # Encode all rows where encoded=FALSE
    python encode_pairs.py

    # Smoke-test with a small batch first
    python encode_pairs.py --limit 200

    # Wipe pseudo_scene_memory and re-encode from scratch
    # (use when changing the PE backbone)
    python encode_pairs.py --force-reencode

    # Use a different PE model
    python encode_pairs.py --encoder PE-Core-L14-336

WHAT IT WRITES
--------------
For each row in raw_scene_pairs (where encoded=FALSE), the script
inserts one row into pseudo_scene_memory with:
    - same label, category, raw_caption
    - rp_caption: text wrapped in "Normal Scene: ..." or "Anomalous Scene: ..."
    - embedding: 1024-D PE features (4096 bytes raw float32)
        anomalous embeddings are pre-multiplied by alpha (0.95) per SAP
    - embed_dim: 1024 (or whatever your PE backbone outputs)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB       = os.getenv("POSTGRES_DB",   "agentvigil_db")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── Defaults ────────────────────────────────────────────────────────
# L14-336 is sized to fit on a 12 GB GPU alongside VideoLLaMA3-2B.
# G14-448 is the paper's flagship choice (87.3 AUC on UCF-Crime) but
# needs ~6 GB and won't fit alongside the VLM on a 3090.
# Both produce 1024-D embeddings; switching is a one-line change.
DEFAULT_ENCODER = "PE-Core-L14-336"
DEFAULT_ALPHA   = 0.95     # paper finds 0.90-1.00 stable, 0.95 best


# ── Schema for pseudo_scene_memory ──────────────────────────────────

PSM_DDL = """
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


# ── Repulsive Prompting wrapper (paper §3.2) ────────────────────────

NORMAL_TEMPLATE = (
    "Normal Scene:\n"
    "  Action Category: {category}\n"
    "  Description: {text}"
)
ANOMALOUS_TEMPLATE = (
    "Anomalous Scene:\n"
    "  Action Category: {category}\n"
    "  Description: {text}"
)


def apply_rp(label: int, category: str, text: str) -> str:
    """Wrap a caption with Repulsive Prompting templates."""
    if label == 0:
        return NORMAL_TEMPLATE.format(category=category.strip(), text=text.strip())
    if label == 1:
        return ANOMALOUS_TEMPLATE.format(category=category.strip(), text=text.strip())
    raise ValueError(f"label must be 0 or 1, got {label}")


# ── PerceptionEncoder loader ────────────────────────────────────────
#
# IMPORTANT — INSTALL NOTE
# The repo is cloned + pip-installed locally. The import path is
#     core.vision_encoder.pe
# NOT
#     perception_models
#
# Repo install steps (run once, in a fresh conda env):
#     git clone https://github.com/facebookresearch/perception_models.git
#     cd perception_models
#     conda create -n perception_models python=3.12 -y
#     conda activate perception_models
#     pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers \
#         --index-url https://download.pytorch.org/whl/cu124
#     conda install ffmpeg -c conda-forge -y
#     pip install torchcodec==0.1 --index-url=https://download.pytorch.org/whl/cu124
#     pip install -e .
#
# Verify before running this script:
#     python -c "import core.vision_encoder.pe as pe; print(pe.CLIP.available_configs())"

class PEWrapper:
    """
    Minimal wrapper around Meta's PerceptionEncoder for text encoding only.
    We don't need video encoding here — only captions get encoded during
    memory construction. The runtime filter (flashback_filter.py) does the
    video encoding online, on a different code path.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.embed_dim = None

    def load(self):
        try:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms
        except ImportError as e:
            sys.exit(
                "ERROR: Could not import 'core.vision_encoder.pe'.\n"
                f"  ImportError: {e}\n\n"
                "  This usually means perception_models is not installed correctly.\n"
                "  Note: the import is `core.vision_encoder.pe`, NOT `perception_models`.\n\n"
                "  Fix:\n"
                "    git clone https://github.com/facebookresearch/perception_models.git\n"
                "    cd perception_models\n"
                "    pip install -e .\n\n"
                "  Then verify:\n"
                "    python -c 'import core.vision_encoder.pe as pe; print(\"OK\")'"
            )

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Confirm the requested config is available
        avail = pe.CLIP.available_configs()
        if self.model_name not in avail:
            sys.exit(
                f"ERROR: Unknown PE config '{self.model_name}'.\n"
                f"  Available: {avail}"
            )

        print(f"[PE] Loading {self.model_name} on {self.device}...")
        # The factory returns a CPU model; we move it to the device manually.
        # Weights download from HuggingFace on first run (~1.7 GB for L14-336).
        self.model = pe.CLIP.from_config(self.model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Tokenizer lives in `transforms`, NOT in `pe`. Reads context_length
        # from the loaded model (e.g. 32 for L14-336).
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Discover embedding dim with a single dummy forward
        with torch.no_grad():
            dummy_tokens = self.tokenizer(["test"]).to(self.device)
            feats = self.model.encode_text(dummy_tokens)
            self.embed_dim = int(feats.shape[-1])
        print(f"[PE] Ready. Embedding dim: {self.embed_dim}")

    def encode_texts(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        """Encode -> L2-normalise -> return float32 (N, D)."""
        import torch
        all_feats = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy().astype(np.float32))
        return np.concatenate(all_feats, axis=0)


# ── Postgres helpers ────────────────────────────────────────────────

def ensure_psm_schema(pg_conn) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(PSM_DDL)


def fetch_pending(pg_conn, limit: int = None) -> List[Tuple[int, int, str, str]]:
    """Return [(raw_id, label, category, raw_caption), ...] for encoded=FALSE."""
    sql = """
        SELECT id, label, category, raw_caption
        FROM   raw_scene_pairs
        WHERE  encoded = FALSE
        ORDER  BY id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    with pg_conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()


def mark_encoded(pg_conn, ids: List[int]) -> None:
    if not ids:
        return
    with pg_conn.cursor() as cur:
        cur.execute(
            "UPDATE raw_scene_pairs SET encoded = TRUE WHERE id = ANY(%s)",
            (ids,),
        )


def insert_encoded(
    pg_conn,
    rows_4tuple: List[Tuple[int, int, str, str]],
    rp_texts:    List[str],
    feats:       np.ndarray,
    embed_dim:   int,
) -> int:
    """Bulk insert into pseudo_scene_memory. Returns rowcount of new inserts."""
    from psycopg2.extras import execute_values

    insert_rows = []
    for (raw_id, label, cat, raw), rp_text, vec in zip(rows_4tuple, rp_texts, feats):
        insert_rows.append((
            int(label),
            cat,
            raw,
            rp_text,
            vec.astype(np.float32, copy=False).tobytes(),
            embed_dim,
        ))

    with pg_conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO pseudo_scene_memory
                (label, category, raw_caption, rp_caption, embedding, embed_dim)
            VALUES %s
            ON CONFLICT (label, rp_caption) DO NOTHING
            """,
            insert_rows,
            page_size=500,
        )
        return cur.rowcount


def stats(pg_conn) -> dict:
    out = {}
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE encoded),
                COUNT(*) FILTER (WHERE NOT encoded)
            FROM raw_scene_pairs
        """)
        r = cur.fetchone()
        out["raw_total"]   = int(r[0])
        out["raw_encoded"] = int(r[1])
        out["raw_pending"] = int(r[2])

        cur.execute("""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE label = 0),
                   COUNT(*) FILTER (WHERE label = 1),
                   COUNT(DISTINCT category)
            FROM pseudo_scene_memory WHERE enabled = TRUE
        """)
        r = cur.fetchone()
        out["psm_total"]      = int(r[0])
        out["psm_normal"]     = int(r[1])
        out["psm_anomalous"]  = int(r[2])
        out["psm_categories"] = int(r[3])
    return out


def force_reencode_setup(pg_conn) -> None:
    """Wipe pseudo_scene_memory + reset encoded flags for clean rebuild."""
    print("[ENC] --force-reencode: wiping pseudo_scene_memory and "
          "resetting encoded flags...")
    with pg_conn.cursor() as cur:
        cur.execute("TRUNCATE pseudo_scene_memory RESTART IDENTITY")
        cur.execute("UPDATE raw_scene_pairs SET encoded = FALSE")


# ── Main ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 2: PE encoding + RP + SAP, no API calls"
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows encoded this run (None = all pending).")
    ap.add_argument("--force-reencode", action="store_true",
                    help="Wipe pseudo_scene_memory and re-encode every raw row.")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help=f"SAP scale factor (default {DEFAULT_ALPHA}).")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER,
                    help=f"PE backbone (default {DEFAULT_ENCODER}).")
    ap.add_argument("--encode-batch", type=int, default=256,
                    help="Texts per PE forward pass.")
    ap.add_argument("--db-batch", type=int, default=512,
                    help="Rows fetched + inserted per loop iteration.")
    args = ap.parse_args()

    try:
        import psycopg2  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install psycopg2-binary")
    import psycopg2

    print(f"[ENC] Connecting to {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}...")
    pg = psycopg2.connect(POSTGRES_URL)
    pg.autocommit = True
    ensure_psm_schema(pg)

    if args.force_reencode:
        force_reencode_setup(pg)

    # Fail fast if nothing to do — saves a slow PE load
    s = stats(pg)
    print(f"[ENC] Initial state:")
    print(f"  raw_scene_pairs:     {s['raw_total']} total, "
          f"{s['raw_pending']} pending, {s['raw_encoded']} already encoded")
    print(f"  pseudo_scene_memory: {s['psm_total']} rows "
          f"({s['psm_normal']}n / {s['psm_anomalous']}a, "
          f"{s['psm_categories']} categories)")

    if s["raw_pending"] == 0:
        print("\n[ENC] Nothing to encode. Run import_pairs_to_postgres.py first, "
              "or use --force-reencode to rebuild.")
        return 0

    target = min(s["raw_pending"], args.limit) if args.limit else s["raw_pending"]
    print(f"[ENC] Will encode: {target} rows")
    print(f"[ENC] Encoder: {args.encoder}")
    print(f"[ENC] SAP alpha: {args.alpha}")
    print()

    # Load PE
    pe = PEWrapper(args.encoder)
    pe.load()
    print()

    # Loop in DB-batches
    total_encoded = 0
    start = time.time()

    while True:
        if args.limit and total_encoded >= args.limit:
            break

        remaining = (args.limit - total_encoded) if args.limit else args.db_batch
        batch_size = min(args.db_batch, remaining)
        rows = fetch_pending(pg, limit=batch_size)
        if not rows:
            break

        rp_texts = [
            apply_rp(label=lbl, category=cat, text=raw)
            for (_id, lbl, cat, raw) in rows
        ]
        labels = np.array([lbl for (_id, lbl, _c, _r) in rows], dtype=np.int8)

        # PE encode
        feats = pe.encode_texts(rp_texts, batch_size=args.encode_batch)

        # Scaled Anomaly Penalisation — anomalous embeddings shrunk
        anom_mask = (labels == 1)
        if anom_mask.any():
            feats[anom_mask] = feats[anom_mask] * float(args.alpha)
        feats = feats.astype(np.float32, copy=False)

        # Insert into pseudo_scene_memory
        inserted = insert_encoded(pg, rows, rp_texts, feats, pe.embed_dim)

        # Flip encoded=TRUE on source rows
        ids = [r[0] for r in rows]
        mark_encoded(pg, ids)

        total_encoded += len(rows)
        elapsed = time.time() - start
        rate    = total_encoded / max(elapsed, 1)
        eta_min = (target - total_encoded) / max(rate, 0.1) / 60
        print(f"[ENC] +{len(rows)} encoded, +{inserted} new memory rows  "
              f"(total {total_encoded}/{target})  "
              f"{rate:.0f} rows/s  ETA {eta_min:.1f} min")

    print()
    final = stats(pg)
    print("[ENC] Done.")
    print(f"  Rows encoded this run:     {total_encoded}")
    print(f"  raw_scene_pairs pending:   {final['raw_pending']}")
    print(f"  pseudo_scene_memory rows:  {final['psm_total']}  "
          f"({final['psm_normal']}n / {final['psm_anomalous']}a)")
    print(f"  Categories represented:    {final['psm_categories']}")
    print(f"  Wall time:                 {(time.time() - start) / 60:.1f} min")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
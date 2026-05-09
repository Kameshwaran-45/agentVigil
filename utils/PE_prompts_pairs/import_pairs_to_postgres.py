"""
import_pairs_to_postgres.py
==============================
Reads scene_pairs.jsonl (created by generate_scene_pairs.py) and
inserts every row into the raw_scene_pairs table in PostgreSQL.

This script does NOT compute embeddings. That's the GPU stage.
This is just the parking step: raw text -> structured DB rows.

REQUIREMENTS
------------
    pip install python-dotenv psycopg2-binary

SETUP
-----
The .env file in the same folder must contain:

    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=agentvigil_db
    POSTGRES_USER=user
    POSTGRES_PASSWORD=password

USAGE
-----
    # Default: read scene_pairs.jsonl from current folder
    python import_pairs_to_postgres.py

    # Or specify a different JSONL file
    python import_pairs_to_postgres.py --input my_pairs.jsonl

    # Re-run is safe — duplicates are silently skipped (UNIQUE constraint)

WHAT HAPPENS
------------
    1. Connects to Postgres using credentials from .env
    2. Creates raw_scene_pairs table if missing (idempotent)
    3. Reads JSONL line by line
    4. Bulk-inserts in batches of 1000 rows
    5. Reports total inserted vs skipped
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
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

RAW_PAIRS_DDL = """
CREATE TABLE IF NOT EXISTS raw_scene_pairs (
    id           SERIAL PRIMARY KEY,
    batch_id     TEXT NOT NULL,
    label        SMALLINT NOT NULL CHECK (label IN (0,1)),
    category     TEXT NOT NULL,
    raw_caption  TEXT NOT NULL,
    encoded      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (label, raw_caption)
);
CREATE INDEX IF NOT EXISTS idx_rsp_encoded ON raw_scene_pairs(encoded);
CREATE INDEX IF NOT EXISTS idx_rsp_label   ON raw_scene_pairs(label);
CREATE INDEX IF NOT EXISTS idx_rsp_batch   ON raw_scene_pairs(batch_id);
"""


def ensure_schema(pg_conn) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(RAW_PAIRS_DDL)


def existing_count(pg_conn) -> dict:
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE label = 0),
                   COUNT(*) FILTER (WHERE label = 1)
            FROM raw_scene_pairs
        """)
        r = cur.fetchone()
    return {"total": int(r[0]), "normal": int(r[1]), "anomalous": int(r[2])}


def parse_jsonl(path: Path) -> List[Tuple[str, int, str, str]]:
    """Return list of (batch_id, label, category, raw_caption) tuples."""
    rows = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                lbl = int(r["label"])
                cat = r["category"].strip()
                txt = r["text"].strip()
                bid = r.get("batch_id", "import")
            except (json.JSONDecodeError, KeyError, ValueError, AttributeError):
                bad_lines += 1
                continue
            if lbl not in (0, 1) or not cat or not txt:
                bad_lines += 1
                continue
            rows.append((bid, lbl, cat, txt))
    if bad_lines:
        print(f"[IMPORT] Skipped {bad_lines} malformed line(s)")
    return rows


def bulk_insert(pg_conn, rows: List[Tuple], page_size: int = 1000) -> int:
    """Bulk insert with ON CONFLICT DO NOTHING. Returns rowcount of new inserts."""
    if not rows:
        return 0
    from psycopg2.extras import execute_values
    inserted = 0
    with pg_conn.cursor() as cur:
        for i in range(0, len(rows), page_size):
            chunk = rows[i:i + page_size]
            execute_values(
                cur,
                """
                INSERT INTO raw_scene_pairs (batch_id, label, category, raw_caption)
                VALUES %s
                ON CONFLICT (label, raw_caption) DO NOTHING
                """,
                chunk,
                page_size=page_size,
            )
            inserted += cur.rowcount
            print(f"[IMPORT] {min(i + page_size, len(rows)):>6}/{len(rows)} processed "
                  f"({inserted} new)")
    return inserted


def main() -> int:
    ap = argparse.ArgumentParser(description="Import JSONL pairs into Postgres")
    ap.add_argument("--input", default=str(SCRIPT_DIR / "scene_pairs.jsonl"),
                    help="Input JSONL file (default: scene_pairs.jsonl)")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"ERROR: Input file not found: {in_path}")

    try:
        import psycopg2  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install psycopg2-binary")

    import psycopg2

    print(f"[IMPORT] Source:      {in_path}")
    print(f"[IMPORT] Destination: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    print()

    # Parse first — fail fast if file is unreadable
    print("[IMPORT] Reading JSONL...")
    rows = parse_jsonl(in_path)
    if not rows:
        sys.exit("ERROR: No valid rows found in JSONL file")
    print(f"[IMPORT] Found {len(rows)} valid rows in source file")

    # Connect + insert
    start = time.time()
    pg = psycopg2.connect(POSTGRES_URL)
    pg.autocommit = True
    ensure_schema(pg)

    before = existing_count(pg)
    print(f"[IMPORT] DB state before: {before['total']} rows "
          f"({before['normal']} normal, {before['anomalous']} anomalous)")
    print()

    inserted = bulk_insert(pg, rows)

    after = existing_count(pg)
    elapsed = time.time() - start

    print()
    print("[IMPORT] Done.")
    print(f"  Source rows:       {len(rows)}")
    print(f"  New rows inserted: {inserted}")
    print(f"  Duplicates skipped:{len(rows) - inserted}")
    print(f"  DB state now:      {after['total']} rows "
          f"({after['normal']} normal, {after['anomalous']} anomalous)")
    print(f"  Wall time:         {elapsed:.1f}s")
    print()
    print("  All rows have encoded=FALSE — next step is the GPU stage:")
    print("    python encode_pairs.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
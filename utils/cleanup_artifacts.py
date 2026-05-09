"""
cleanup_phase1_artifacts.py
==============================
Removes the Phase-1 CLIP prompts table from PostgreSQL.

WHAT THIS DROPS
---------------
    clip_prompts        — the 19 hardcoded CLIP anomaly/normal prompts
                          (replaced in Phase 2 by pseudo_scene_memory
                           which is generated via LLM at any scale)

WHAT THIS DOES NOT TOUCH
------------------------
    chunks              — your video processing logbook (KEEP)
    videos              — per-video summaries (KEEP)
    keywords            — Tier 1 routing keywords (KEEP)
    raw_scene_pairs     — Phase 2 raw caption staging (KEEP)
    pseudo_scene_memory — Phase 2 PE-encoded memory (KEEP, when created)

    Milvus collection agentvigil_captions — completely separate system,
    not touched by this script. (KEEP)

REQUIRES CONFIRMATION
---------------------
This is destructive. The script:
  1. Counts what's there
  2. Prints what will be dropped
  3. Asks for explicit "yes" confirmation
  4. Drops only after confirmation

REQUIREMENTS
------------
    pip install python-dotenv psycopg2-binary

USAGE
-----
    python cleanup_phase1_artifacts.py

    # Force without confirmation (use with care):
    python cleanup_phase1_artifacts.py --yes

    # Dry-run — show what would happen, no changes:
    python cleanup_phase1_artifacts.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

# Tables this script may drop. Hard-coded so no other tables get touched.
DEPRECATED_TABLES = ["clip_prompts"]


def table_exists(pg_conn, name: str) -> bool:
    with pg_conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = %s
        """, (name,))
        return cur.fetchone() is not None


def row_count(pg_conn, name: str) -> int:
    with pg_conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {name}")
        return int(cur.fetchone()[0])


def survey_keep_tables(pg_conn) -> dict:
    """Show row counts for tables we're keeping — confirms nothing was lost."""
    keep = ["chunks", "videos", "keywords", "raw_scene_pairs", "pseudo_scene_memory"]
    out = {}
    for t in keep:
        if table_exists(pg_conn, t):
            try:
                out[t] = row_count(pg_conn, t)
            except Exception:
                out[t] = "(error reading)"
        else:
            out[t] = "(does not exist)"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Drop deprecated Phase-1 CLIP tables")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would happen, change nothing")
    args = ap.parse_args()

    try:
        import psycopg2  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pip install psycopg2-binary")

    import psycopg2

    print(f"[CLEANUP] Connecting to {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}...")
    pg = psycopg2.connect(POSTGRES_URL)
    pg.autocommit = True

    # Show kept tables first — reassures user nothing important is at risk
    print()
    print("[CLEANUP] Tables that WILL be kept (untouched):")
    for tname, cnt in survey_keep_tables(pg).items():
        print(f"    {tname:25} {cnt}")

    # Find the deprecated tables that actually exist
    print()
    print("[CLEANUP] Phase-1 deprecated tables:")
    to_drop = []
    for t in DEPRECATED_TABLES:
        if table_exists(pg, t):
            try:
                cnt = row_count(pg, t)
                to_drop.append((t, cnt))
                print(f"    {t:25} {cnt} rows  ->  WILL BE DROPPED")
            except Exception as e:
                print(f"    {t:25} (error: {e})")
        else:
            print(f"    {t:25} (already removed)")

    if not to_drop:
        print()
        print("[CLEANUP] Nothing to do. All deprecated tables are already gone.")
        return 0

    if args.dry_run:
        print()
        print("[CLEANUP] --dry-run: stopping here. No tables were dropped.")
        return 0

    if not args.yes:
        print()
        prompt = "[CLEANUP] Type 'yes' to drop these tables: "
        try:
            answer = input(prompt).strip().lower()
        except KeyboardInterrupt:
            print("\n[CLEANUP] Aborted.")
            return 1
        if answer != "yes":
            print("[CLEANUP] Aborted — no tables were dropped.")
            return 1

    # Do it
    print()
    with pg.cursor() as cur:
        for tname, cnt in to_drop:
            print(f"[CLEANUP] DROP TABLE {tname} ({cnt} rows)...")
            cur.execute(f"DROP TABLE IF EXISTS {tname} CASCADE")

    print()
    print("[CLEANUP] Done.")
    print()
    print("[CLEANUP] Tables remaining (post-cleanup state):")
    for tname, cnt in survey_keep_tables(pg).items():
        print(f"    {tname:25} {cnt}")
    print()
    print("  Code cleanup (optional, do later):")
    print("    rm clip_filter.py clip_prompt_store.py")
    print("    Remove CLIP_* constants from config.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
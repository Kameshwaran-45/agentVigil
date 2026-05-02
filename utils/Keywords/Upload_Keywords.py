"""
scripts/upload_keywords.py
===========================
Connects to the running PostgreSQL container and seeds the keywords table.
Run AFTER generate_keywords.py has produced keywords_seed.sql.

USAGE
-----
    # Using DB URL from .env / config.py (default)
    python scripts/upload_keywords.py

    # Override connection inline
    python scripts/upload_keywords.py \
        --host localhost --port 5432 \
        --db agentvigil --user postgres --password postgres

    # Dry-run: print what would be inserted without writing
    python scripts/upload_keywords.py --dry-run

    # Force re-generate before uploading
    python scripts/upload_keywords.py --regenerate

WHAT IT DOES
------------
1. Creates the keywords table if it doesn't exist
2. Upserts all rows (safe to re-run; existing weights are updated)
3. Prints a per-category count confirmation
4. Optionally shows rows that changed weight vs what was already in DB
"""

import argparse
import os
import sys

# Allow running from repo root or from scripts/

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def get_conn(args):
    import psycopg2
    # Prefer explicit CLI args; fall back to config.py values
    if args.host:
        dsn = (
            f"host={args.host} port={args.port} "
            f"dbname={args.db} user={args.user} password={args.password}"
        )
    else:
        try:
            from config import POSTGRES_URL
            dsn = POSTGRES_URL
        except ImportError:
            print("[ERROR] Could not import config.py. Pass --host / --user / --password manually.")
            sys.exit(1)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def ensure_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS keywords (
            id               SERIAL PRIMARY KEY,
            category         TEXT NOT NULL,
            keyword          TEXT NOT NULL,
            weight           INTEGER NOT NULL DEFAULT 1,
            is_high_priority BOOLEAN NOT NULL DEFAULT FALSE,
            created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (category, keyword)
        );
        CREATE INDEX IF NOT EXISTS idx_kw_category ON keywords(category);
        CREATE INDEX IF NOT EXISTS idx_kw_keyword  ON keywords(keyword);
    """)


def upload(conn, records, dry_run=False):
    from psycopg2.extras import execute_values

    if dry_run:
        print(f"  [DRY-RUN] Would upsert {len(records)} rows — no DB changes made.")
        return 0, 0

    with conn.cursor() as cur:
        ensure_table(cur)

        # Fetch existing to report changes
        cur.execute("SELECT category, keyword, weight FROM keywords")
        existing = {(r[0], r[1]): r[2] for r in cur.fetchall()}

        data = [
            (r["category"], r["keyword"], r["weight"], r["is_high_priority"])
            for r in records
        ]
        execute_values(cur, """
            INSERT INTO keywords (category, keyword, weight, is_high_priority)
            VALUES %s
            ON CONFLICT (category, keyword) DO UPDATE
              SET weight           = EXCLUDED.weight,
                  is_high_priority = EXCLUDED.is_high_priority
        """, data)

        # Count new vs updated
        new_rows     = sum(1 for r in records if (r["category"], r["keyword"]) not in existing)
        updated_rows = sum(
            1 for r in records
            if (r["category"], r["keyword"]) in existing
            and existing[(r["category"], r["keyword"])] != r["weight"]
        )
        return new_rows, updated_rows


def print_db_summary(conn):
    from psycopg2.extras import RealDictCursor
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT category,
                   COUNT(*)                                           AS total,
                   SUM(CASE WHEN weight = 3 THEN 1 ELSE 0 END)       AS primary_kw,
                   SUM(CASE WHEN weight = 2 THEN 1 ELSE 0 END)       AS action_kw,
                   SUM(CASE WHEN weight = 1 THEN 1 ELSE 0 END)       AS context_kw,
                   MAX(is_high_priority::int)::boolean                AS high_priority
            FROM keywords
            GROUP BY category
            ORDER BY category
        """)
        rows = cur.fetchall()

    print(f"\n{'─'*72}")
    print(f"  {'Category':<28} {'Total':>5}  {'P(3)':>4} {'A(2)':>4} {'C(1)':>4}  {'HighPri':>7}")
    print(f"{'─'*72}")
    grand = 0
    for r in rows:
        hp = "⚡ YES" if r["high_priority"] else "  no "
        print(
            f"  {r['category']:<28} {r['total']:>5}  "
            f"{r['primary_kw']:>4} {r['action_kw']:>4} {r['context_kw']:>4}  {hp}"
        )
        grand += r["total"]
    print(f"{'─'*72}")
    print(f"  {'TOTAL':<28} {grand:>5}")
    print(f"{'─'*72}\n")


def main():
    parser = argparse.ArgumentParser(description="Upload keyword seed to PostgreSQL")
    parser.add_argument("--host",       default="", help="DB host (default: from config.py)")
    parser.add_argument("--port",       default="5432")
    parser.add_argument("--db",         default="agentvigil")
    parser.add_argument("--user",       default="postgres")
    parser.add_argument("--password",   default="postgres")
    parser.add_argument("--dry-run",    action="store_true", help="Print plan, no DB writes")
    parser.add_argument("--regenerate", action="store_true", help="Re-run generate_keywords.py first")
    args = parser.parse_args()

    if args.regenerate:
        print("[INFO] Regenerating seed files...")
        import subprocess
        subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "generate_keywords.py")],
            check=True,
        )

    # Import records from generate module (avoids re-parsing the SQL file)
    sys.path.insert(0, SCRIPT_DIR)
    from Generate_Keywords import build_records
    records = build_records()

    print(f"[UPLOAD] {len(records)} keywords → PostgreSQL")
    if args.dry_run:
        print("[UPLOAD] DRY-RUN mode — no changes will be made\n")

    try:
        conn = get_conn(args)
    except Exception as e:
        print(f"[ERROR] Cannot connect to PostgreSQL: {e}")
        print("  Make sure your container is up:  docker ps | grep postgres")
        sys.exit(1)

    new_rows, updated_rows = upload(conn, records, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"[UPLOAD] Done — {new_rows} new rows, {updated_rows} weights updated")
        print_db_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
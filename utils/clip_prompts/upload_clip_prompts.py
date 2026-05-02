"""
scripts/upload_clip_prompts.py
================================
Connects to the running PostgreSQL container and seeds the clip_prompts table.
Run AFTER generate_clip_prompts.py.

USAGE
-----
    # Uses connection from config.py / .env (default)
    python scripts/upload_clip_prompts.py

    # Override connection
    python scripts/upload_clip_prompts.py \
        --host localhost --port 5432 \
        --db agentvigil --user postgres --password postgres

    # Dry-run — print plan without writing
    python scripts/upload_clip_prompts.py --dry-run

    # Disable all existing prompts first, then insert fresh set
    python scripts/upload_clip_prompts.py --reset

    # Re-generate seed files then upload in one step
    python scripts/upload_clip_prompts.py --regenerate

IMPORTANT
---------
The clip_filter.py CLIPFilter must call reload_prompts() after this
script runs for changes to take effect in a running server.
Or just restart Streamlit.
"""

import argparse, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def get_conn(args):
    import psycopg2
    if args.host:
        dsn = (f"host={args.host} port={args.port} "
               f"dbname={args.db} user={args.user} password={args.password}")
    else:
        try:
            from config import POSTGRES_URL
            dsn = POSTGRES_URL
        except ImportError:
            print("[ERROR] Could not import config.py. Pass --host etc. manually.")
            sys.exit(1)
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    return conn


def ensure_table(cur):
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clip_prompts (
            id          SERIAL PRIMARY KEY,
            prompt_type TEXT    NOT NULL,
            category    TEXT,
            prompt_text TEXT    NOT NULL,
            enabled     BOOLEAN NOT NULL DEFAULT TRUE,
            weight      REAL    NOT NULL DEFAULT 1.0,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (prompt_type, prompt_text)
        );
        CREATE INDEX IF NOT EXISTS idx_cp_type     ON clip_prompts(prompt_type);
        CREATE INDEX IF NOT EXISTS idx_cp_category ON clip_prompts(category);
        CREATE INDEX IF NOT EXISTS idx_cp_enabled  ON clip_prompts(enabled);
    """)


def upload(conn, records, dry_run=False, reset=False):
    from psycopg2.extras import execute_values

    if dry_run:
        anomaly = sum(1 for r in records if r["prompt_type"] == "anomaly")
        normal  = sum(1 for r in records if r["prompt_type"] == "normal")
        print(f"  [DRY-RUN] Would upsert {len(records)} rows "
              f"({anomaly} anomaly, {normal} normal) — no DB changes.")
        return 0, 0

    with conn.cursor() as cur:
        ensure_table(cur)

        if reset:
            cur.execute("UPDATE clip_prompts SET enabled = FALSE")
            print(f"  [RESET] Disabled all existing prompts")

        existing_q = "SELECT prompt_type, prompt_text FROM clip_prompts"
        cur.execute(existing_q)
        existing = {(r[0], r[1]) for r in cur.fetchall()}

        data = [
            (r["prompt_type"], r["category"], r["prompt_text"],
             r["enabled"], r["weight"])
            for r in records
        ]
        execute_values(cur, """
            INSERT INTO clip_prompts
                (prompt_type, category, prompt_text, enabled, weight)
            VALUES %s
            ON CONFLICT (prompt_type, prompt_text) DO UPDATE
              SET category = EXCLUDED.category,
                  enabled  = EXCLUDED.enabled,
                  weight   = EXCLUDED.weight
        """, data)

        new_rows = sum(
            1 for r in records
            if (r["prompt_type"], r["prompt_text"]) not in existing
        )
        updated = len(records) - new_rows
        return new_rows, updated


def print_db_summary(conn):
    from psycopg2.extras import RealDictCursor
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                prompt_type,
                COUNT(*)                                              AS total,
                SUM(CASE WHEN enabled THEN 1 ELSE 0 END)             AS enabled_count,
                COUNT(DISTINCT category)                              AS categories
            FROM clip_prompts
            GROUP BY prompt_type
            ORDER BY prompt_type
        """)
        rows = cur.fetchall()

        cur.execute("""
            SELECT category, COUNT(*) AS n
            FROM clip_prompts
            WHERE prompt_type = 'anomaly' AND enabled = TRUE
            GROUP BY category
            ORDER BY category
        """)
        cat_rows = cur.fetchall()

    print(f"\n{'─'*60}")
    print(f"  clip_prompts table — current state")
    print(f"{'─'*60}")
    for r in rows:
        print(f"  {r['prompt_type']:<10s}  total={r['total']:>3}  "
              f"enabled={r['enabled_count']:>3}  categories={r['categories']}")
    print(f"  {'─'*50}")
    print(f"  Anomaly breakdown by category:")
    for r in cat_rows:
        cat = r["category"] or "_general"
        print(f"    {cat:<28s} {r['n']:>3} prompts")
    print(f"{'─'*60}\n")
    print("  ✅ CLIP filter will use these prompts on next load/reload.")
    print("  → Restart Streamlit OR call clip_filter.reload_prompts()\n")


def main():
    parser = argparse.ArgumentParser(description="Upload CLIP prompts to PostgreSQL")
    parser.add_argument("--host",       default="")
    parser.add_argument("--port",       default="5432")
    parser.add_argument("--db",         default="agentvigil")
    parser.add_argument("--user",       default="postgres")
    parser.add_argument("--password",   default="postgres")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--reset",      action="store_true",
                        help="Disable all existing prompts before inserting")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-run generate_clip_prompts.py first")
    args = parser.parse_args()

    if args.regenerate:
        import subprocess
        print("[INFO] Regenerating seed files...")
        subprocess.run(
            [sys.executable,
             os.path.join(SCRIPT_DIR, "generate_clip_prompts.py")],
            check=True,
        )

    sys.path.insert(0, SCRIPT_DIR)
    from generate_clip_prompts import build_records
    records = build_records()

    print(f"[UPLOAD] {len(records)} CLIP prompts → PostgreSQL")

    try:
        conn = get_conn(args)
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        print("  docker ps | grep postgres  — is the container running?")
        sys.exit(1)

    new_rows, updated = upload(conn, records, dry_run=args.dry_run, reset=args.reset)

    if not args.dry_run:
        print(f"[UPLOAD] Done — {new_rows} new, {updated} updated")
        print_db_summary(conn)

    conn.close()


if __name__ == "__main__":
    main()
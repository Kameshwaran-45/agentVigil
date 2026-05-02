"""
clip_prompt_store.py
=====================
Runtime CLIP prompt registry backed by the PostgreSQL clip_prompts table.

WHAT IT REPLACES
----------------
    CLIP_ANOMALY_PROMPTS  — hardcoded list in config.py (12 items)
    CLIP_NORMAL_PROMPTS   — hardcoded list in config.py (7 items)

Both are now loaded from the DB, enabling prompt management without
code changes or server restarts.

CACHING + RE-ENCODING LIFECYCLE
---------------------------------
    CLIPPromptStore is created once inside CLIPFilter.load().
    Text features are encoded immediately on first load and cached.

    When you add/edit prompts in the DB:
      → call clip_filter.reload_prompts()
      → this calls store.reload() which re-fetches + re-encodes
      → zero downtime; the old features are used until reload completes

INTERFACE  (used only by CLIPFilter — not called by app.py directly)
----------------------------------------------------------------------
    store = CLIPPromptStore(pg_conn)
    store.anomaly_texts   → List[str]   (enabled anomaly prompts)
    store.normal_texts    → List[str]   (enabled normal prompts)
    store.reload()        → re-fetch from DB
    store.stats()         → dict with counts
"""

from __future__ import annotations
from typing import List, Dict


class CLIPPromptStore:
    """
    In-memory CLIP prompt cache loaded from clip_prompts table.
    Pass the live psycopg2 connection from DatabaseManager.
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn
        self.anomaly_texts: List[str] = []
        self.normal_texts:  List[str] = []
        self._rows: List[Dict] = []
        self.load()

    # ── LOAD / RELOAD ───────────────────────────────────────────────

    def load(self) -> None:
        """
        Fetch all ENABLED prompts from DB.
        Falls back to hardcoded config lists if the table is empty
        (e.g. first run before upload_clip_prompts.py has been run).
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT prompt_type, category, prompt_text, weight
                    FROM   clip_prompts
                    WHERE  enabled = TRUE
                    ORDER  BY prompt_type, category NULLS LAST, prompt_text
                """)
                rows = cur.fetchall()
        except Exception as e:
            print(f"[CLIPPromptStore] WARNING: DB read failed ({e}). "
                  "Falling back to config.py defaults.")
            rows = []

        if rows:
            self._rows = [
                {"type": r[0], "category": r[1],
                 "text": r[2], "weight": r[3]}
                for r in rows
            ]
            self.anomaly_texts = [r["text"] for r in self._rows if r["type"] == "anomaly"]
            self.normal_texts  = [r["text"] for r in self._rows if r["type"] == "normal"]
            print(f"[CLIPPromptStore] Loaded {len(self.anomaly_texts)} anomaly + "
                  f"{len(self.normal_texts)} normal prompts from DB")
        else:
            # ── Fallback: config.py hardcoded lists ─────────────────
            print("[CLIPPromptStore] Table empty — using config.py defaults. "
                  "Run: python scripts/upload_clip_prompts.py")
            try:
                from config import CLIP_ANOMALY_PROMPTS, CLIP_NORMAL_PROMPTS
                self.anomaly_texts = list(CLIP_ANOMALY_PROMPTS)
                self.normal_texts  = list(CLIP_NORMAL_PROMPTS)
            except ImportError:
                self.anomaly_texts = []
                self.normal_texts  = []
            self._rows = []
            print(f"[CLIPPromptStore] Fallback: {len(self.anomaly_texts)} anomaly, "
                  f"{len(self.normal_texts)} normal")

    def reload(self) -> None:
        """Re-fetch from DB. Call after adding/editing prompts."""
        print("[CLIPPromptStore] Reloading prompts from DB...")
        self.load()

    # ── STATS ────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        from collections import Counter
        cat_counts = Counter(
            r["category"] or "_general"
            for r in self._rows if r["type"] == "anomaly"
        )
        return {
            "anomaly_count":    len(self.anomaly_texts),
            "normal_count":     len(self.normal_texts),
            "total":            len(self.anomaly_texts) + len(self.normal_texts),
            "source":           "database" if self._rows else "config_fallback",
            "anomaly_by_category": dict(cat_counts),
        }
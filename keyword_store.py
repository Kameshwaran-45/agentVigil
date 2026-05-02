"""
keyword_store.py
=================
Runtime keyword registry backed by PostgreSQL.

WHAT IT REPLACES
----------------
These two objects that were hardcoded in config.py and imported by agent.py:

    ALL_CRIME_KEYWORDS  — flat list, used for total hit count
    EVENT_KEYWORDS      — {category: [kw, ...]} dict, used for per-category scoring

WHY DB-BACKED
-------------
- Keywords can be added, removed, or re-weighted via SQL without
  touching Python code or restarting the server.
- Weight column gives the router a richer signal than raw count.
- scripts/upload_keywords.py is the single place to change keywords.

CACHING
-------
Keywords are loaded ONCE from DB when KeywordStore() is first called,
then kept in memory for the lifetime of the process.  There is no
per-chunk DB round-trip.

To reload after a DB change (e.g. you added new keywords mid-session):
    store.reload()

INTERFACE (drop-in replacement for config.py dicts)
----------------------------------------------------
    store = KeywordStore(db_conn)

    # Replaces:  sum(1 for kw in ALL_CRIME_KEYWORDS if kw in caption)
    store.count_hits(caption)                    -> int (raw hit count)

    # Weighted version — better routing signal
    store.weighted_hits(caption)                 -> float

    # Replaces:  EVENT_KEYWORDS[category]
    store.keywords_for(category)                 -> List[str]

    # Replaces:  for event_type, keywords in EVENT_KEYWORDS.items()
    store.all_categories()                       -> List[str]

    # Replaces:  for e in ["Robbery", "Arson", ...] (high-priority check)
    store.is_high_priority(category)             -> bool

    # Full dict — for code that still iterates EVENT_KEYWORDS directly
    store.as_event_keywords_dict()               -> {category: [kw, ...]}
    store.as_flat_list()                         -> [kw, ...]
"""

from __future__ import annotations
from typing import Dict, List, Optional


class KeywordStore:
    """
    In-memory keyword registry loaded from the keywords table.

    Pass the live psycopg2 connection from DatabaseManager:
        store = KeywordStore(db.pg_conn)
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn
        self._rows: List[Dict] = []          # raw rows from DB
        self._by_category: Dict[str, List[Dict]] = {}   # {cat: [{keyword, weight}, ...]}
        self._flat: List[str] = []           # all keywords, deduplicated
        self._high_priority: set[str] = set()
        self._loaded = False
        self.load()

    # ── LOAD / RELOAD ───────────────────────────────────────────────

    def load(self) -> None:
        """Load all keywords from DB into memory."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    SELECT category, keyword, weight, is_high_priority
                    FROM keywords
                    ORDER BY category, weight DESC, keyword
                """)
                rows = cur.fetchall()
        except Exception as e:
            print(f"[KeywordStore] WARNING: could not load from DB ({e}). "
                  "Falling back to empty store — keyword routing disabled.")
            rows = []

        self._rows = [
            {"category": r[0], "keyword": r[1], "weight": r[2], "is_high_priority": r[3]}
            for r in rows
        ]

        self._by_category = {}
        seen_flat: set[str] = set()
        self._high_priority = set()

        for row in self._rows:
            cat = row["category"]
            self._by_category.setdefault(cat, []).append(
                {"keyword": row["keyword"], "weight": row["weight"]}
            )
            seen_flat.add(row["keyword"])
            if row["is_high_priority"]:
                self._high_priority.add(cat)

        self._flat = sorted(seen_flat)
        self._loaded = True

        total = len(self._rows)
        cats  = len(self._by_category)
        print(f"[KeywordStore] Loaded {total} keywords across {cats} categories from DB")

    def reload(self) -> None:
        """Refresh from DB (call after adding/editing keywords via SQL)."""
        print("[KeywordStore] Reloading keywords from DB...")
        self.load()

    # ── QUERY INTERFACE ─────────────────────────────────────────────

    def count_hits(self, caption: str) -> int:
        """
        Raw keyword hit count across ALL categories.
        Drop-in replacement for:
            sum(1 for kw in ALL_CRIME_KEYWORDS if kw in caption_lower)
        """
        cl = caption.lower()
        return sum(1 for kw in self._flat if kw in cl)

    def weighted_hits(self, caption: str) -> float:
        """
        Weighted hit count — sum of weights for matched keywords.
        Better routing signal than raw count.
        Use this instead of count_hits() for the complexity score.
        """
        cl = caption.lower()
        total = 0.0
        for row in self._rows:
            if row["keyword"] in cl:
                total += row["weight"]
        return total

    def matched_categories(self, caption: str) -> List[str]:
        """
        Return categories that have at least one keyword hit.
        Drop-in replacement for the matched_events loop in agent.py.
        """
        cl = caption.lower()
        matched = []
        for cat, entries in self._by_category.items():
            if any(e["keyword"] in cl for e in entries):
                matched.append(cat)
        return matched

    def best_category(self, caption: str) -> tuple[str, int]:
        """
        Return (best_category, hit_count) — used by Tier 1 scoring.
        Weighted: category with highest sum-of-weights wins.
        Drop-in replacement for the Tier 1 loop in agent.py.
        """
        cl = caption.lower()
        best_cat = "Normal_Videos_event"
        best_score = 0
        for cat, entries in self._by_category.items():
            score = sum(e["weight"] for e in entries if e["keyword"] in cl)
            if score > best_score:
                best_score = score
                best_cat = cat
        return best_cat, best_score

    def keywords_for(self, category: str) -> List[str]:
        """Return plain keyword list for a given category."""
        return [e["keyword"] for e in self._by_category.get(category, [])]

    def all_categories(self) -> List[str]:
        """Return all category names present in the DB."""
        return list(self._by_category.keys())

    def is_high_priority(self, category: str) -> bool:
        """True if this category forces Tier-3 routing."""
        return category in self._high_priority

    def has_high_priority_match(self, matched_categories: List[str]) -> bool:
        """True if any matched category is high-priority."""
        return any(self.is_high_priority(c) for c in matched_categories)

    # ── COMPATIBILITY HELPERS ───────────────────────────────────────

    def as_event_keywords_dict(self) -> Dict[str, List[str]]:
        """
        {category: [keyword, ...]}
        For code that still iterates EVENT_KEYWORDS directly.
        """
        return {cat: self.keywords_for(cat) for cat in self._by_category}

    def as_flat_list(self) -> List[str]:
        """
        Flat deduplicated keyword list.
        Equivalent to ALL_CRIME_KEYWORDS.
        """
        return list(self._flat)

    # ── STATS ────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "total_keywords": len(self._rows),
            "categories": len(self._by_category),
            "high_priority_categories": sorted(self._high_priority),
            "per_category": {
                cat: len(entries)
                for cat, entries in self._by_category.items()
            },
        }
"""
Database Layer — Stage 1 Storage (Structured Event Store + Vector Index)
=========================================================================
WHAT:  Manages the dual-database knowledge base from the architecture:
       - PostgreSQL = "Structured Event Store" in the diagram
       - Milvus     = "Vector Index" in the diagram

WHY TWO DATABASES:
  PostgreSQL answers: "Show all robberies this week" (structured SQL)
  Milvus answers:     "Find clips SIMILAR to this robbery" (vector cosine)

  Together they enable the Adaptive Decision Router:
    Tier 1 → PostgreSQL keyword queries (fast, cheap)
    Tier 2 → Milvus semantic search (finds meaning, not just words)
    Tier 3 → Both DBs feed context to the LLM for deep reasoning

THE PROACTIVE LOOP:
  1. Every chunk → stored in BOTH databases immediately
  2. Agent queries both DBs to find patterns
  3. Patterns trigger alerts before humans notice them

CONNECTS TO: perception.py provides captions to store
             agent.py queries both DBs for reasoning
             app.py reads alerts and stats for display
"""

import json
from typing import Dict, List, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from pymilvus import (
    connections, Collection, FieldSchema,
    CollectionSchema, DataType, utility,
)

from config import (
    POSTGRES_URL, MILVUS_HOST, MILVUS_PORT,
    MILVUS_COLLECTION, EMBEDDING_DIM, SIMILAR_INCIDENT_TOP_K,
)


class DatabaseManager:
    def __init__(self):
        self.pg_conn = None
        self.milvus_collection = None
        self.connected = False

    # ── CONNECTION ──────────────────────────────────────────────────

    def connect(self):
        self._connect_postgres()
        self._connect_milvus()
        self.connected = True

    def _connect_postgres(self):
        self.pg_conn = psycopg2.connect(POSTGRES_URL)
        self.pg_conn.autocommit = True
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id              SERIAL PRIMARY KEY,
                    camera_id       TEXT NOT NULL DEFAULT 'CAM-01',
                    video_name      TEXT NOT NULL,
                    chunk_index     INTEGER NOT NULL,
                    start_sec       REAL,
                    end_sec         REAL,
                    num_frames      INTEGER,
                    vlm_caption     TEXT,
                    event_type      TEXT DEFAULT 'Normal Activity',
                    anomaly_score   REAL DEFAULT 0.0,
                    severity        TEXT DEFAULT 'LOW',
                    alert_status    TEXT DEFAULT 'NORMAL',
                    tier_used       INTEGER DEFAULT 1,
                    agent_narrative TEXT,
                    agent_evidence  TEXT,
                    recommendation  TEXT,
                    similar_count   INTEGER DEFAULT 0,
                    metadata        JSONB DEFAULT '{}',
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(camera_id, video_name, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS videos (
                    id               SERIAL PRIMARY KEY,
                    camera_id        TEXT NOT NULL DEFAULT 'CAM-01',
                    video_name       TEXT NOT NULL,
                    duration_sec     REAL,
                    total_chunks     INTEGER,
                    anomalous_chunks INTEGER DEFAULT 0,
                    primary_event    TEXT DEFAULT 'Normal Activity',
                    overall_score    REAL DEFAULT 0.0,
                    overall_severity TEXT DEFAULT 'LOW',
                    overall_alert    TEXT DEFAULT 'NORMAL',
                    narrative        TEXT,
                    processed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(camera_id, video_name)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_camera ON chunks(camera_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_video  ON chunks(video_name);
                CREATE INDEX IF NOT EXISTS idx_chunks_event  ON chunks(event_type);
                CREATE INDEX IF NOT EXISTS idx_chunks_alert  ON chunks(alert_status);
                CREATE INDEX IF NOT EXISTS idx_chunks_score  ON chunks(anomaly_score DESC);
                CREATE INDEX IF NOT EXISTS idx_videos_camera ON videos(camera_id);

                -- Keyword store table (seeded by scripts/upload_keywords.py)
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

                -- CLIP prompt store (seeded by scripts/upload_clip_prompts.py)
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
        print("[DB] PostgreSQL ready")

    def _connect_milvus(self):
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(MILVUS_COLLECTION):
            fields = [
                FieldSchema("id",          DataType.INT64,         is_primary=True, auto_id=True),
                FieldSchema("pg_id",       DataType.INT64),
                # Scalar metadata for filtering and camera-boost only.
                # event_type and anomaly_score are intentionally NOT stored here —
                # they are written/corrected by update_chunk_analysis() in Postgres
                # AFTER the agent finishes. search_similar() always fetches the
                # authoritative values from Postgres via a single batched query.
                FieldSchema("camera_id",   DataType.VARCHAR, max_length=64),
                FieldSchema("video_name",  DataType.VARCHAR, max_length=256),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("embedding",   DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            ]
            schema = CollectionSchema(fields, "AgentVigil caption embeddings")
            self.milvus_collection = Collection(MILVUS_COLLECTION, schema)
            self.milvus_collection.create_index(
                "embedding",
                {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
            )
        else:
            self.milvus_collection = Collection(MILVUS_COLLECTION)

        self.milvus_collection.load()
        count = self.milvus_collection.num_entities
        print(f"[DB] Milvus ready | {count} vectors stored")

    # ── STORE ───────────────────────────────────────────────────────

    def store_chunk(self, video_name, chunk_index, start_sec, end_sec,
                    num_frames, caption, event_type, embedding,
                    anomaly_score=0.0, metadata=None, camera_id="CAM-01") -> int:
        """Store in BOTH databases. Returns PostgreSQL row ID."""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (camera_id, video_name, chunk_index, start_sec, end_sec,
                    num_frames, vlm_caption, event_type, anomaly_score, metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (camera_id, video_name, chunk_index) DO UPDATE SET
                    vlm_caption=EXCLUDED.vlm_caption, event_type=EXCLUDED.event_type,
                    anomaly_score=EXCLUDED.anomaly_score, metadata=EXCLUDED.metadata
                RETURNING id
            """, (camera_id, video_name, chunk_index, start_sec, end_sec, num_frames,
                  caption, event_type, anomaly_score, json.dumps(metadata or {})))
            pg_id = cur.fetchone()[0]

        self.milvus_collection.insert([
            [pg_id], [camera_id], [video_name], [chunk_index], [embedding]
        ])
        self.milvus_collection.flush()

        return pg_id

    def store_video_summary(self, video_name, duration_sec, total_chunks,
                            anomalous_chunks, primary_event, overall_score,
                            overall_severity, overall_alert, narrative,
                            camera_id="CAM-01"):
        """Store video-level summary after full analysis."""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO videos (camera_id, video_name, duration_sec, total_chunks,
                    anomalous_chunks, primary_event, overall_score,
                    overall_severity, overall_alert, narrative)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (camera_id, video_name) DO UPDATE SET
                    duration_sec=EXCLUDED.duration_sec, total_chunks=EXCLUDED.total_chunks,
                    anomalous_chunks=EXCLUDED.anomalous_chunks, primary_event=EXCLUDED.primary_event,
                    overall_score=EXCLUDED.overall_score, overall_severity=EXCLUDED.overall_severity,
                    overall_alert=EXCLUDED.overall_alert, narrative=EXCLUDED.narrative
            """, (camera_id, video_name, duration_sec, total_chunks, anomalous_chunks,
                  primary_event, overall_score, overall_severity, overall_alert, narrative))

    # ── RETRIEVE ────────────────────────────────────────────────────

    def search_similar(self, query_embedding, top_k=SIMILAR_INCIDENT_TOP_K,
                       exclude_video=None, camera_id=None) -> List[Dict]:
        """
        Milvus vector search — find similar past incidents by caption embedding.

        HOW IT WORKS (two-step):
          Step 1 — Milvus cosine search: finds the top-K most semantically similar
                   past captions using the 384-dim embedding as the query.
                   Milvus stores only the vector + lightweight scalar fields needed
                   for filtering/boosting (pg_id, camera_id, video_name,
                   chunk_index). Caption text and final verdict (anomaly_score,
                   alert_status) are NOT stored in Milvus — they live in Postgres
                   as the authoritative source.

          Step 2 — Postgres enrichment (single batched query): takes all pg_ids
                   returned by Milvus and fetches the final agent verdict for each
                   (vlm_caption, anomaly_score, alert_status, event_type) in one
                   SELECT … WHERE id = ANY(%s). This is how Tier 2 "learns from
                   the outcome" of past events — it reads the score written by
                   update_chunk_analysis(), not the initial 0.0 placeholder.

        NOTE: chunks from the current video being analysed are excluded via
        exclude_video so the agent cannot treat its own in-progress chunks as
        historical evidence.

        If camera_id is provided, same-camera results receive a +0.1 similarity
        boost (capped at 1.0) to prioritise location-relevant history.
        """
        if self.milvus_collection.num_entities == 0:
            return []

        self.milvus_collection.flush()

        # ── Step 1: Milvus cosine search ────────────────────────────
        filters = []
        if exclude_video:
            filters.append(f'video_name != "{exclude_video}"')
        expr = " && ".join(filters) if filters else None

        try:
            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k * 2,  # fetch extra so camera-boost reordering works
                expr=expr,
                output_fields=["pg_id", "camera_id", "video_name", "chunk_index"],
            )
        except Exception as e:
            print(f"[DB] Milvus search failed: {e}")
            return []

        # Collect raw hits (pg_id → raw similarity + camera metadata).
        # Use hit.fields (plain dict) instead of hit.entity.get() —
        # newer Milvus SDK (2.3+) dropped the default-value argument from
        # hit.entity.get(), so hit.entity.get("key", default) raises TypeError.
        # hit.fields is always a plain Python dict and works across all versions.
        raw_hits: List[Dict] = []
        for hits in results:
            for hit in hits:
                fields = hit.fields if hasattr(hit, "fields") else dict(hit.entity)
                raw_hits.append({
                    "pg_id":       fields.get("pg_id"),
                    "hit_camera":  fields.get("camera_id", ""),
                    "video_name":  fields.get("video_name"),
                    "chunk_index": fields.get("chunk_index"),
                    "raw_score":   round(hit.score, 4),
                })

        if not raw_hits:
            return []

        # ── Step 2: Postgres enrichment — single batched query ───────
        # Fetch the final agent verdict for all matching pg_ids at once.
        # anomaly_score and alert_status reflect what update_chunk_analysis()
        # wrote after the agent finished — NOT the initial 0.0 placeholder.
        pg_ids = [h["pg_id"] for h in raw_hits if h["pg_id"] is not None]
        pg_lookup: Dict[int, Dict] = {}
        if pg_ids:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, vlm_caption, event_type, anomaly_score, alert_status
                    FROM   chunks
                    WHERE  id = ANY(%s)
                    """,
                    (pg_ids,),
                )
                for row in cur.fetchall():
                    pg_lookup[row["id"]] = dict(row)

        # ── Assemble + apply camera boost ────────────────────────────
        similar: List[Dict] = []
        for h in raw_hits:
            pg_id    = h["pg_id"]
            pg_data  = pg_lookup.get(pg_id, {})
            score    = h["raw_score"]
            same_cam = camera_id is not None and h["hit_camera"] == camera_id

            if same_cam:
                score = min(score + 0.1, 1.0)

            similar.append({
                "similarity":    round(score, 4),
                "same_camera":   same_cam,
                "camera_id":     h["hit_camera"],
                "video_name":    h["video_name"],
                "chunk_index":   h["chunk_index"],
                # From Postgres — authoritative values post-agent-verdict:
                "event_type":    pg_data.get("event_type", "Unknown"),
                "caption":       pg_data.get("vlm_caption", ""),
                "anomaly_score": pg_data.get("anomaly_score", 0.0),
                "alert_status":  pg_data.get("alert_status", "NORMAL"),
            })

        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:top_k]

    def keyword_search(self, keywords: List[str], limit=20) -> List[Dict]:
        """PostgreSQL keyword search — Tier 1 retrieval."""
        conditions = " OR ".join(["vlm_caption ILIKE %s" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM chunks WHERE {conditions}
                ORDER BY anomaly_score DESC LIMIT %s
            """, params + [limit])
            return [dict(r) for r in cur.fetchall()]

    def get_video_chunks(self, video_name: str, camera_id: str = None) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            if camera_id:
                cur.execute(
                    "SELECT * FROM chunks WHERE video_name=%s AND camera_id=%s ORDER BY chunk_index",
                    (video_name, camera_id))
            else:
                cur.execute(
                    "SELECT * FROM chunks WHERE video_name=%s ORDER BY chunk_index",
                    (video_name,))
            return [dict(r) for r in cur.fetchall()]

    def get_alerts(self, limit=50, camera_id: str = None) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            if camera_id:
                cur.execute("""
                    SELECT * FROM chunks WHERE alert_status IN ('ALERT','WATCH')
                    AND camera_id=%s
                    ORDER BY anomaly_score DESC, created_at DESC LIMIT %s
                """, (camera_id, limit))
            else:
                cur.execute("""
                    SELECT * FROM chunks WHERE alert_status IN ('ALERT','WATCH')
                    ORDER BY anomaly_score DESC, created_at DESC LIMIT %s
                """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_all_videos(self, camera_id: str = None) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            if camera_id:
                cur.execute(
                    "SELECT * FROM videos WHERE camera_id=%s ORDER BY processed_at DESC",
                    (camera_id,))
            else:
                cur.execute("SELECT * FROM videos ORDER BY processed_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_cameras(self) -> List[str]:
        """Return all distinct camera IDs that have data."""
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT DISTINCT camera_id FROM videos ORDER BY camera_id")
            return [r[0] for r in cur.fetchall()]

    def get_event_stats(self, camera_id: str = None) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            if camera_id:
                cur.execute("""
                    SELECT event_type, COUNT(*) as count,
                           ROUND(AVG(anomaly_score)::numeric, 3) as avg_score,
                           SUM(CASE WHEN alert_status='ALERT' THEN 1 ELSE 0 END) as alerts
                    FROM chunks WHERE camera_id=%s
                    GROUP BY event_type ORDER BY count DESC
                """, (camera_id,))
            else:
                cur.execute("""
                    SELECT event_type, COUNT(*) as count,
                           ROUND(AVG(anomaly_score)::numeric, 3) as avg_score,
                           SUM(CASE WHEN alert_status='ALERT' THEN 1 ELSE 0 END) as alerts
                    FROM chunks GROUP BY event_type ORDER BY count DESC
                """)
            return [dict(r) for r in cur.fetchall()]

    def reset_data(self, camera_id: str = None) -> Dict:
        """
        Clear stored analysis data.
        If camera_id given: wipe only that camera's rows.
        If None: wipe ALL chunks + videos + Milvus collection.
        Keywords and CLIP prompts are always preserved.
        """
        with self.pg_conn.cursor() as cur:
            if camera_id:
                cur.execute("DELETE FROM chunks WHERE camera_id=%s", (camera_id,))
                pg_chunks = cur.rowcount
                cur.execute("DELETE FROM videos WHERE camera_id=%s", (camera_id,))
                pg_videos = cur.rowcount
            else:
                cur.execute("DELETE FROM chunks")
                pg_chunks = cur.rowcount
                cur.execute("DELETE FROM videos")
                pg_videos = cur.rowcount

        # Rebuild Milvus collection from scratch
        from pymilvus import utility
        collection_name = self.milvus_collection.name
        self.milvus_collection.release()
        utility.drop_collection(collection_name)

        # Recreate with updated schema (no event_type — see _connect_milvus)
        fields = [
            FieldSchema("id",          DataType.INT64,         is_primary=True, auto_id=True),
            FieldSchema("pg_id",       DataType.INT64),
            FieldSchema("camera_id",   DataType.VARCHAR, max_length=64),
            FieldSchema("video_name",  DataType.VARCHAR, max_length=256),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("embedding",   DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ]
        schema = CollectionSchema(fields, "AgentVigil caption embeddings")
        self.milvus_collection = Collection(collection_name, schema)
        self.milvus_collection.create_index(
            "embedding",
            {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
        )
        self.milvus_collection.load()

        scope = f"camera {camera_id}" if camera_id else "all cameras"
        print(f"[DB] Reset complete ({scope}): {pg_chunks} chunks, {pg_videos} videos deleted. Milvus rebuilt.")
        return {"pg_chunks_deleted": pg_chunks, "pg_videos_deleted": pg_videos, "scope": scope}

    def update_chunk_analysis(self, pg_id, anomaly_score, severity, alert_status,
                               tier_used, narrative, evidence, recommendation, similar_count):
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                UPDATE chunks SET anomaly_score=%s, severity=%s, alert_status=%s,
                    tier_used=%s, agent_narrative=%s, agent_evidence=%s,
                    recommendation=%s, similar_count=%s
                WHERE id=%s
            """, (anomaly_score, severity, alert_status, tier_used,
                  narrative, evidence, recommendation, similar_count, pg_id))

    def _get_chunk_by_id(self, pg_id) -> Optional[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM chunks WHERE id=%s", (pg_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_clip_prompts(self, enabled_only: bool = True) -> list:
        """
        Return clip_prompts rows for CLIPPromptStore.
        Called once at startup — cached in CLIPPromptStore.
        """
        with self.pg_conn.cursor() as cur:
            if enabled_only:
                cur.execute("""
                    SELECT prompt_type, category, prompt_text, weight
                    FROM   clip_prompts
                    WHERE  enabled = TRUE
                    ORDER  BY prompt_type, category NULLS LAST, prompt_text
                """)
            else:
                cur.execute("""
                    SELECT prompt_type, category, prompt_text, weight
                    FROM   clip_prompts
                    ORDER  BY prompt_type, category NULLS LAST, prompt_text
                """)
            return cur.fetchall()

    def get_all_keywords(self) -> list:
        """
        Return all rows from the keywords table.
        Called once at startup by KeywordStore — no per-chunk overhead.
        Returns list of (category, keyword, weight, is_high_priority) tuples.
        """
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                SELECT category, keyword, weight, is_high_priority
                FROM keywords
                ORDER BY category, weight DESC, keyword
            """)
            return cur.fetchall()

    def keyword_count(self) -> int:
        """How many keywords are currently in the DB."""
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM keywords")
            return cur.fetchone()[0]

    def close(self):
        if self.pg_conn: self.pg_conn.close()
        try: connections.disconnect("default")
        except: pass
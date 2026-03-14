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
                    UNIQUE(video_name, chunk_index)
                );

                CREATE TABLE IF NOT EXISTS videos (
                    id              SERIAL PRIMARY KEY,
                    video_name      TEXT UNIQUE NOT NULL,
                    duration_sec    REAL,
                    total_chunks    INTEGER,
                    anomalous_chunks INTEGER DEFAULT 0,
                    primary_event   TEXT DEFAULT 'Normal Activity',
                    overall_score   REAL DEFAULT 0.0,
                    overall_severity TEXT DEFAULT 'LOW',
                    overall_alert   TEXT DEFAULT 'NORMAL',
                    narrative       TEXT,
                    processed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_name);
                CREATE INDEX IF NOT EXISTS idx_chunks_event ON chunks(event_type);
                CREATE INDEX IF NOT EXISTS idx_chunks_alert ON chunks(alert_status);
                CREATE INDEX IF NOT EXISTS idx_chunks_score ON chunks(anomaly_score DESC);
            """)
        print("[DB] PostgreSQL ready")

    def _connect_milvus(self):
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(MILVUS_COLLECTION):
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("pg_id", DataType.INT64),
                FieldSchema("video_name", DataType.VARCHAR, max_length=256),
                FieldSchema("chunk_index", DataType.INT64),
                FieldSchema("event_type", DataType.VARCHAR, max_length=128),
                FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
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
                    anomaly_score=0.0, metadata=None) -> int:
        """Store in BOTH databases. Returns PostgreSQL row ID."""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chunks (video_name, chunk_index, start_sec, end_sec,
                    num_frames, vlm_caption, event_type, anomaly_score, metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (video_name, chunk_index) DO UPDATE SET
                    vlm_caption=EXCLUDED.vlm_caption, event_type=EXCLUDED.event_type,
                    anomaly_score=EXCLUDED.anomaly_score, metadata=EXCLUDED.metadata
                RETURNING id
            """, (video_name, chunk_index, start_sec, end_sec, num_frames,
                  caption, event_type, anomaly_score, json.dumps(metadata or {})))
            pg_id = cur.fetchone()[0]

        self.milvus_collection.insert([
            [pg_id], [video_name], [chunk_index], [event_type], [embedding]
        ])
        self.milvus_collection.flush()

        return pg_id

    def store_video_summary(self, video_name, duration_sec, total_chunks,
                            anomalous_chunks, primary_event, overall_score,
                            overall_severity, overall_alert, narrative):
        """Store video-level summary after full analysis."""
        with self.pg_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO videos (video_name, duration_sec, total_chunks,
                    anomalous_chunks, primary_event, overall_score,
                    overall_severity, overall_alert, narrative)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (video_name) DO UPDATE SET
                    duration_sec=EXCLUDED.duration_sec, total_chunks=EXCLUDED.total_chunks,
                    anomalous_chunks=EXCLUDED.anomalous_chunks, primary_event=EXCLUDED.primary_event,
                    overall_score=EXCLUDED.overall_score, overall_severity=EXCLUDED.overall_severity,
                    overall_alert=EXCLUDED.overall_alert, narrative=EXCLUDED.narrative
            """, (video_name, duration_sec, total_chunks, anomalous_chunks,
                  primary_event, overall_score, overall_severity, overall_alert, narrative))

    # ── RETRIEVE ────────────────────────────────────────────────────

    def search_similar(self, query_embedding, top_k=SIMILAR_INCIDENT_TOP_K,
                       exclude_video=None) -> List[Dict]:
        """Milvus vector search — find similar past incidents."""

        # Guard: can't search an empty collection
        if self.milvus_collection.num_entities == 0:
            return []

        # Flush to make sure recent inserts are searchable
        self.milvus_collection.flush()

        expr = f'video_name != "{exclude_video}"' if exclude_video else None

        try:
            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=top_k,
                expr=expr,
                output_fields=["pg_id", "video_name", "chunk_index", "event_type"],
            )
        except Exception as e:
            print(f"[DB] Milvus search failed: {e}")
            return []

        similar = []
        for hits in results:
            for hit in hits:
                pg_id = hit.entity.get("pg_id")
                pg_data = self._get_chunk_by_id(pg_id)
                similar.append({
                    "similarity": round(hit.score, 4),
                    "video_name": hit.entity.get("video_name"),
                    "chunk_index": hit.entity.get("chunk_index"),
                    "event_type": hit.entity.get("event_type"),
                    "caption": pg_data.get("vlm_caption", "") if pg_data else "",
                    "anomaly_score": pg_data.get("anomaly_score", 0) if pg_data else 0,
                })
        return similar

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

    def get_video_chunks(self, video_name: str) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM chunks WHERE video_name=%s ORDER BY chunk_index",
                (video_name,))
            return [dict(r) for r in cur.fetchall()]

    def get_alerts(self, limit=50) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM chunks WHERE alert_status IN ('ALERT','WATCH')
                ORDER BY anomaly_score DESC, created_at DESC LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_all_videos(self) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM videos ORDER BY processed_at DESC")
            return [dict(r) for r in cur.fetchall()]

    def get_event_stats(self) -> List[Dict]:
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT event_type, COUNT(*) as count,
                       ROUND(AVG(anomaly_score)::numeric, 3) as avg_score,
                       SUM(CASE WHEN alert_status='ALERT' THEN 1 ELSE 0 END) as alerts
                FROM chunks GROUP BY event_type ORDER BY count DESC
            """)
            return [dict(r) for r in cur.fetchall()]

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

    def close(self):
        if self.pg_conn: self.pg_conn.close()
        try: connections.disconnect("default")
        except: pass
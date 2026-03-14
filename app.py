"""
AgentVigil — Streamlit Dashboard
==================================
Full pipeline: Chunk → CLIP Filter → VLM Caption → Store → Agent

Uses EXACT same chunking as benchmark_vlm.py to ensure production
results match benchmark quality.
"""

import os
import time
import tempfile
import streamlit as st
import pandas as pd

from config import (
    PRIMARY_VLM_NAME, CLIP_ENABLED,
    CHUNK_OVERLAP_SEC, SAMPLING_STRATEGY,
)
from video_processor import extract_chunks_and_frames, get_video_info
from perception import PerceptionEngine
from embeddings import EmbeddingEngine
from database import DatabaseManager
from agent import AdaptiveReasoningAgent
from clip_filter import CLIPFilter


st.set_page_config(
    page_title="AgentVigil",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric > div { padding: 8px; }
    div[data-testid="stExpander"] details summary p { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# SYSTEM INIT (cached — loads once)
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_system():
    embedder = EmbeddingEngine()
    db = DatabaseManager()
    db.connect()
    perception = PerceptionEngine()
    perception.load()
    clip_filter = CLIPFilter()
    clip_filter.load()
    agent = AdaptiveReasoningAgent(db, embedder)
    return perception, embedder, db, agent, clip_filter


def severity_icon(sev):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚪")


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("# 🛡️ AgentVigil")
    st.sidebar.caption("AI-Powered Surveillance Intelligence")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigate",
        [
            "📹 Process Video",
            "🚨 Alerts",
            "🔍 Search",
            "📊 Stats",
            "📋 History",
        ],
    )

    st.sidebar.divider()
    st.sidebar.markdown("**System Config**")

    overlap_mode = "Sliding Window" if CHUNK_OVERLAP_SEC > 0 else "Hard Cuts (Benchmark)"
    st.sidebar.code(
        f"VLM: {PRIMARY_VLM_NAME}\n"
        f"CLIP Filter: {'ON' if CLIP_ENABLED else 'OFF'}\n"
        f"Chunking: {overlap_mode}\n"
        f"Overlap: {CHUNK_OVERLAP_SEC}s\n"
        f"Sampling: {SAMPLING_STRATEGY}\n"
        f"Mode: IMAGE | Quant: 4-bit",
        language=None,
    )

    st.sidebar.markdown("**Pipeline**")
    st.sidebar.markdown("""
    ```
    Video
     → Adaptive Chunking
     → CLIP Pre-Filter
     → LLaVA-OV Caption
     → PostgreSQL + Milvus
     → Adaptive Router
       ├── T1: Keyword
       ├── T2: Semantic
       └── T3: Temporal
     → Alert Generation
    ```
    """)

    return page


# ═════════════════════════════════════════════════════════════════════
# PAGE: PROCESS VIDEO
# ═════════════════════════════════════════════════════════════════════

def page_process(perception, embedder, db, agent, clip_filter):
    st.title("📹 Process Surveillance Video")
    st.markdown(
        "**Pipeline:** Adaptive Chunk → CLIP Filter → "
        "VLM Caption → Store → Agent Analyze"
    )

    uploaded = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"]
    )
    if not uploaded:
        st.info("Upload a surveillance video to begin.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    video_name = os.path.splitext(uploaded.name)[0]
    info = get_video_info(tmp_path)

    # Video info metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Duration", f"{info['duration_sec']}s")
    c2.metric("FPS", f"{info['fps']}")
    c3.metric("Resolution", f"{info['width']}×{info['height']}")
    c4.metric("Chunk Size", f"{info['chunk_duration']}s")
    c5.metric("Stride", f"{info['stride_sec']}s")
    c6.metric("Est. Chunks", f"{info['total_chunks']}")

    st.video(tmp_path)

    if not st.button(
        "🚀 Run AgentVigil Pipeline",
        type="primary",
        use_container_width=True,
    ):
        return

    run_pipeline(
        tmp_path, video_name, info,
        perception, embedder, db, agent, clip_filter,
    )


def run_pipeline(
    video_path, video_name, info,
    perception, embedder, db, agent, clip_filter,
):
    pipeline_start = time.time()

    # ── STAGE 0: Adaptive Chunking (benchmark-identical) ────────────
    with st.status(
        "⏳ Stage 0: Adaptive Chunking & Frame Extraction...",
        expanded=False,
    ) as s0:
        output_dir = tempfile.mkdtemp()
        chunks = extract_chunks_and_frames(video_path, output_dir)
        total_frames = sum(c["num_frames"] for c in chunks.values())
        first_chunk = next(iter(chunks.values()))

        overlap_label = (
            f"overlap={first_chunk['overlap_sec']}s"
            if first_chunk["overlap_sec"] > 0
            else "hard cuts"
        )
        s0.update(
            label=(
                f"✅ Stage 0: {len(chunks)} chunks × "
                f"{first_chunk['chunk_duration_sec']}s "
                f"({overlap_label}) | {total_frames} frames"
            ),
            state="complete",
        )

    # ── CLIP PRE-FILTER ─────────────────────────────────────────────
    with st.status(
        "⏳ CLIP Pre-Filter: Scoring chunks...", expanded=False
    ) as s_clip:
        passed_chunks, skipped_chunks, clip_stats = (
            clip_filter.filter_chunks(chunks)
        )
        if clip_stats["enabled"]:
            s_clip.update(
                label=(
                    f"✅ CLIP Filter: {clip_stats['passed']}/"
                    f"{clip_stats['total']} passed "
                    f"({clip_stats['compute_saved_pct']}% filtered)"
                ),
                state="complete",
            )
        else:
            s_clip.update(
                label="⚪ CLIP Filter: Disabled — all chunks pass",
                state="complete",
            )

    # Show CLIP score distribution
    if clip_stats["enabled"] and clip_stats.get("scores"):
        st.markdown("#### CLIP Anomaly Scores")
        scores = clip_stats["scores"]

        score_df = pd.DataFrame([
            {
                "Chunk": f"C{k}",
                "CLIP Score": v,
                "Status": "✅ VLM" if k in passed_chunks else "❌ Skip",
            }
            for k, v in sorted(scores.items())
        ])
        st.bar_chart(score_df.set_index("Chunk")["CLIP Score"])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("→ VLM", clip_stats["passed"])
        mc2.metric("Skipped", clip_stats["skipped"])
        mc3.metric("Compute Saved", f"{clip_stats['compute_saved_pct']}%")

    # ── Process SKIPPED chunks (auto Normal, no VLM) ────────────────
    for cidx, cinfo in sorted(skipped_chunks.items()):
        embedding = embedder.embed_text(
            "Normal activity. No anomaly detected."
        )
        db.store_chunk(
            video_name=video_name,
            chunk_index=cidx,
            start_sec=cinfo["start_sec"],
            end_sec=cinfo["end_sec"],
            num_frames=cinfo["num_frames"],
            caption=(
                f"[CLIP filtered — score: "
                f"{cinfo.get('clip_score', 0):.3f}] Normal activity."
            ),
            event_type="Normal Activity",
            embedding=embedding,
            anomaly_score=0.0,
            metadata={
                "clip_score": cinfo.get("clip_score", 0),
                "clip_filtered": True,
            },
        )

    # ── STAGE 1+2: VLM + Agent (passed chunks only) ────────────────
    st.markdown("---")
    st.markdown("### 🧠 Stage 1: VLM Perception + Stage 2: Adaptive Routing")

    total_to_process = len(passed_chunks)
    if total_to_process == 0:
        st.success(
            "All chunks classified as Normal by CLIP filter. "
            "No VLM processing needed."
        )
    else:
        progress = st.progress(0, text="Starting VLM captioning...")
        results_container = st.container()

    tier_counts = {0: len(skipped_chunks), 1: 0, 2: 0, 3: 0}
    alert_count = 0
    chunk_results = []

    for i, (cidx, cinfo) in enumerate(sorted(passed_chunks.items())):
        progress.progress(
            (i + 1) / max(total_to_process, 1),
            text=(
                f"Chunk {cidx} "
                f"({cinfo['start_sec']:.0f}s-{cinfo['end_sec']:.0f}s) | "
                f"{i+1}/{total_to_process}"
            ),
        )

        # Stage 1: VLM Caption
        caption, latency = perception.caption_chunk(cinfo["frame_paths"])
        event_type = perception.extract_event_type(caption)
        embedding = embedder.embed_text(caption)

        # Store in both DBs
        pg_id = db.store_chunk(
            video_name=video_name,
            chunk_index=cidx,
            start_sec=cinfo["start_sec"],
            end_sec=cinfo["end_sec"],
            num_frames=cinfo["num_frames"],
            caption=caption,
            event_type=event_type,
            embedding=embedding,
            metadata={
                "clip_score": cinfo.get("clip_score", 0),
                "clip_filtered": False,
                "latency": latency,
            },
        )

        # Stage 2: Adaptive Decision Router
        analysis = agent.analyze_chunk(
            video_name=video_name,
            chunk_index=cidx,
            caption=caption,
            embedding=embedding,
            pg_id=pg_id,
        )

        tier_counts[analysis["tier_used"]] = (
            tier_counts.get(analysis["tier_used"], 0) + 1
        )
        if analysis["alert_status"] == "ALERT":
            alert_count += 1

        chunk_results.append({
            "chunk_index": cidx,
            **cinfo,
            "caption": caption,
            "event_type": event_type,
            "latency": latency,
            **analysis,
        })

        # Real-time display
        sev = severity_icon(analysis["severity"])
        with results_container.expander(
            f"{sev} Chunk {cidx} | "
            f"{cinfo['start_sec']:.0f}-{cinfo['end_sec']:.0f}s | "
            f"{event_type} | Tier {analysis['tier_used']} | "
            f"Score: {analysis['anomaly_score']} | {latency:.1f}s",
            expanded=(analysis["alert_status"] == "ALERT"),
        ):
            # Show frames (same layout as benchmark output)
            n_show = min(len(cinfo["frame_paths"]), 4)
            frame_cols = st.columns(n_show)
            for j in range(n_show):
                frame_cols[j].image(
                    cinfo["frame_paths"][j],
                    caption=f"Frame {j}",
                    use_container_width=True,
                )

            st.markdown(f"**Caption:** {caption}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Anomaly Score", f"{analysis['anomaly_score']}")
            m2.metric("Tier Used", f"{analysis['tier_used']}")
            m3.metric("Similar Past", f"{analysis['similar_count']}")
            m4.metric("CLIP Score", f"{cinfo.get('clip_score', 'N/A')}")

            if analysis.get("evidence"):
                st.markdown("**Evidence:**")
                st.code(analysis["evidence"], language=None)
            if analysis.get("recommendation"):
                st.info(f"💡 {analysis['recommendation']}")

    if total_to_process > 0:
        progress.progress(1.0, text="✅ All chunks processed!")

    # ── STAGE 3: Full Video Analysis ────────────────────────────────
    with st.status(
        "🤖 Stage 3: Video-level analysis...", expanded=False
    ) as s3:
        video_analysis = agent.analyze_full_video(
            video_name, info["duration_sec"]
        )
        s3.update(
            label="✅ Stage 3: Analysis complete", state="complete"
        )

    pipeline_time = time.time() - pipeline_start

    # ── RESULTS DASHBOARD ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")

    sev_i = severity_icon(video_analysis.get("overall_severity", "LOW"))
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Event", video_analysis.get("primary_event", "Unknown"))
    r2.metric("Score", f"{video_analysis.get('overall_score', 0):.2f}")
    r3.metric(
        "Severity",
        f"{sev_i} {video_analysis.get('overall_severity', 'LOW')}",
    )
    r4.metric(
        "Alert",
        "🚨 YES" if video_analysis.get("overall_alert") == "ALERT"
        else "✅ NO",
    )
    r5.metric(
        "Anomalous",
        f"{video_analysis.get('anomalous_chunks', 0)}/"
        f"{video_analysis.get('total_chunks', 0)}",
    )
    r6.metric("Pipeline Time", f"{pipeline_time:.1f}s")

    # Tier usage breakdown
    st.markdown("### 🔀 Adaptive Router — Tier Usage")
    t0, t1, t2, t3 = st.columns(4)
    t0.metric("T0 CLIP Skip", tier_counts.get(0, 0))
    t1.metric("T1 Keyword", tier_counts.get(1, 0))
    t2.metric("T2 Semantic", tier_counts.get(2, 0))
    t3.metric("T3 Temporal", tier_counts.get(3, 0))

    # Narrative
    st.markdown("### 📝 Agent Narrative")
    st.info(video_analysis.get("narrative", "No narrative."))

    # Chunk timeline table
    if video_analysis.get("chunk_details"):
        st.markdown("### 📋 Chunk Timeline")
        df = pd.DataFrame(video_analysis["chunk_details"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Pipeline performance summary
    st.markdown("### ⚡ Pipeline Performance")
    if chunk_results:
        latencies = [r["latency"] for r in chunk_results]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Avg VLM Latency", f"{sum(latencies)/len(latencies):.1f}s")
        p2.metric("Total VLM Calls", f"{len(chunk_results)}")
        p3.metric(
            "Chunks Skipped (CLIP)",
            f"{len(skipped_chunks)}",
        )
        vlm_time = sum(latencies)
        p4.metric(
            "VLM Compute Saved",
            f"{len(skipped_chunks) * (sum(latencies)/max(len(latencies),1)):.0f}s",
        )


# ═════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ═════════════════════════════════════════════════════════════════════

def page_alerts(db):
    st.title("🚨 Alert Dashboard")
    alerts = db.get_alerts(limit=50)
    if not alerts:
        st.info("No alerts yet. Process a video to generate alerts.")
        return

    # Summary metrics
    alert_count = sum(1 for a in alerts if a["alert_status"] == "ALERT")
    watch_count = sum(1 for a in alerts if a["alert_status"] == "WATCH")
    ac1, ac2 = st.columns(2)
    ac1.metric("🔴 Active Alerts", alert_count)
    ac2.metric("🟡 Watch Items", watch_count)

    st.markdown("---")

    for a in alerts:
        sev = severity_icon(a.get("severity", "LOW"))
        with st.expander(
            f"{sev} {a['event_type']} | {a['video_name']} "
            f"C{a['chunk_index']} | Score: {a['anomaly_score']} | "
            f"Tier {a.get('tier_used', '?')}",
            expanded=(a["alert_status"] == "ALERT"),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{a.get('anomaly_score', 0):.2f}")
            c2.metric(
                "Time",
                f"{a['start_sec']}s-{a['end_sec']}s",
            )
            c3.metric("Tier", f"{a.get('tier_used', '?')}")
            c4.metric("Similar Past", f"{a.get('similar_count', 0)}")

            st.markdown(f"**Caption:** {a['vlm_caption']}")

            if a.get("agent_narrative"):
                st.markdown(f"**Narrative:** {a['agent_narrative']}")
            if a.get("agent_evidence"):
                st.code(a["agent_evidence"], language=None)
            if a.get("recommendation"):
                st.info(f"💡 {a['recommendation']}")


# ═════════════════════════════════════════════════════════════════════
# PAGE: SEARCH
# ═════════════════════════════════════════════════════════════════════

def page_search(db, embedder):
    st.title("🔍 Search Similar Incidents")
    st.markdown(
        "Describe a scenario → Milvus finds similar past incidents "
        "via vector similarity search."
    )

    query = st.text_area(
        "Describe the incident",
        placeholder="e.g., A person grabs a bag and runs away",
        height=100,
    )

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching..."):
            embedding = embedder.embed_text(query)
            results = db.search_similar(embedding, top_k=10)

        if results:
            st.success(f"Found {len(results)} similar incidents")
            for r in results:
                with st.expander(
                    f"📎 {r['event_type']} | {r['video_name']} "
                    f"C{r['chunk_index']} | "
                    f"Similarity: {r['similarity']:.3f}"
                ):
                    st.markdown(f"**Caption:** {r['caption']}")
                    st.metric(
                        "Past Anomaly Score",
                        f"{r.get('anomaly_score', 0):.2f}",
                    )
        else:
            st.warning("No similar incidents found.")


# ═════════════════════════════════════════════════════════════════════
# PAGE: STATS
# ═════════════════════════════════════════════════════════════════════

def page_stats(db):
    st.title("📊 Statistics")
    stats = db.get_event_stats()
    if not stats:
        st.info("No data yet. Process videos to see statistics.")
        return

    df = pd.DataFrame(stats)

    st.markdown("### Event Distribution")
    st.bar_chart(df.set_index("event_type")["count"])

    st.markdown("### Details")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ═════════════════════════════════════════════════════════════════════

def page_history(db):
    st.title("📋 Video History")
    videos = db.get_all_videos()
    if not videos:
        st.info("No videos processed yet.")
        return

    for v in videos:
        sev = severity_icon(v.get("overall_severity", "LOW"))
        with st.expander(
            f"{sev} {v['video_name']} | {v['primary_event']} | "
            f"Score: {v.get('overall_score', 0):.2f} | "
            f"{v.get('anomalous_chunks', 0)}/"
            f"{v.get('total_chunks', 0)} anomalous"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{v.get('duration_sec', 0)}s")
            c2.metric("Chunks", f"{v.get('total_chunks', 0)}")
            c3.metric("Score", f"{v.get('overall_score', 0):.2f}")
            c4.metric("When", str(v.get("processed_at", ""))[:19])

            if v.get("narrative"):
                st.markdown(f"**Narrative:**\n{v['narrative']}")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    try:
        perception, embedder, db, agent, clip_filter = init_system()
    except Exception as e:
        st.error(f"⚠️ System init failed: {e}")
        st.markdown(
            "**Check:** Is PostgreSQL and Milvus running? GPU available?"
        )
        st.code(
            "docker compose up -d\npip install -r requirements.txt",
            language="bash",
        )
        return

    if page == "📹 Process Video":
        page_process(perception, embedder, db, agent, clip_filter)
    elif page == "🚨 Alerts":
        page_alerts(db)
    elif page == "🔍 Search":
        page_search(db, embedder)
    elif page == "📊 Stats":
        page_stats(db)
    elif page == "📋 History":
        page_history(db)


if __name__ == "__main__":
    main()
"""
AgentVigil — Streamlit Dashboard
==================================
Full pipeline: Chunk → CLIP Filter → VLM Caption → Store → Agent

Uses EXACT same chunking as benchmark_vlm.py to ensure production
results match benchmark quality.

CHANGES FROM ORIGINAL:
  - Model and Prompt are now selectable via sidebar dropdowns.
  - init_system() split: core services cached once; perception engine
    cached per model key so switching models loads a new instance
    without restarting the server.
  - caption_chunk() now receives prompt_type from session state.
"""

import os
import re
import time
import html
import tempfile
import streamlit as st
import pandas as pd
from datetime import datetime

from config import (
        CHUNK_CONTEXT_ENABLED,
        CHUNK_CONTEXT_MAX_CHARS,
        CHUNK_OVERLAP_SEC,
        FRAMES_PER_SECOND,
        SAMPLING_STRATEGY,
        VLM_REGISTRY,
        DEFAULT_VLM,
        FLASHBACK_ENABLED,
        FLASHBACK_ENCODER,
        FLASHBACK_TOP_K,
        FLASHBACK_THRESHOLD,
        FLASHBACK_FEED_CAPTIONS_TO_VLM,
        FLASHBACK_VLM_PRIOR_K,
    )
from prompt_loader import get_prompt_registry, get_default_stem
from video_processor import extract_chunks_and_frames, get_video_info
from perception import PerceptionEngine
from embeddings import EmbeddingEngine
from database import DatabaseManager
from agent import AdaptiveReasoningAgent
from flashback_filter import FlashbackFilter


st.set_page_config(
    page_title="AgentVigil",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    :root {
        --brand-cyan: #22d3ee;
        --brand-blue: #2563eb;
        --brand-ink: #04132c;
        --card-bg: rgba(8, 26, 54, 0.82);
        --card-border: rgba(110, 178, 255, 0.28);
        --muted: #98b4d8;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 12% -10%, rgba(34, 211, 238, 0.20), transparent 60%),
            radial-gradient(1000px 500px at 88% 0%, rgba(37, 99, 235, 0.18), transparent 62%),
            linear-gradient(135deg, #020913 0%, #031021 50%, #041731 100%);
    }

    .block-container {
        max-width: 1300px;
        padding-top: 1.2rem;
    }

    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.28);
    }

    .stMetric label {
        color: var(--muted) !important;
        font-weight: 600 !important;
    }

    .av-hero {
        margin: 0 0 0.9rem 0;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        border: 1px solid var(--card-border);
        background: linear-gradient(120deg, rgba(34, 211, 238, 0.10), rgba(37, 99, 235, 0.08));
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
    }

    .av-hero h2 {
        margin: 0;
        font-size: 1.35rem;
        color: #e6f1ff;
        letter-spacing: 0.2px;
    }

    .av-hero p {
        margin: 0.35rem 0 0;
        color: var(--muted);
        font-size: 0.95rem;
    }

    .av-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.65rem;
        margin: 0.65rem 0 0.2rem;
    }

    .av-status-card {
        border-radius: 12px;
        padding: 0.75rem 0.8rem;
        border: 1px solid var(--card-border);
        background: var(--card-bg);
    }

    .av-status-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        font-weight: 700;
    }

    .av-status-value {
        color: #f4fbff;
        font-size: 1.12rem;
        line-height: 1.35;
        font-weight: 700;
        word-break: break-word;
    }

    .av-status-alert {
        border-color: rgba(255, 125, 125, 0.45);
        background: linear-gradient(120deg, rgba(183, 28, 28, 0.34), rgba(90, 16, 16, 0.24));
    }

    .av-status-watch {
        border-color: rgba(255, 196, 84, 0.45);
        background: linear-gradient(120deg, rgba(120, 83, 12, 0.36), rgba(69, 47, 8, 0.25));
    }

    div[data-testid="stExpander"] details {
        border: 1px solid var(--card-border);
        border-radius: 12px;
        background: rgba(7, 20, 43, 0.72);
    }

    div[data-testid="stExpander"] details summary p { font-size: 14px; }

    .av-page-card {
    margin-top: 3rem;
    margin-bottom: 1.5rem;
    padding: 0.5rem 1rem 0.65rem;
    border-radius: 10px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}

.av-page-card h1 {
    margin: 0 0 0.1rem 0;
    font-size: 1.7rem;
    color: #e6f1ff;
    font-weight: 700;
}

.av-page-card .av-subtitle {
    color: var(--muted);
    font-size: 0.76rem;
    margin-bottom: 0.65rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# SYSTEM INIT (cached — loads once per unique key)
# ═════════════════════════════════════════════════════════════════════

@st.cache_resource
def init_core_system():
    """Loads everything that does NOT depend on model choice."""
    embedder = EmbeddingEngine()
    db = DatabaseManager()
    db.connect()
    agent = AdaptiveReasoningAgent(db, embedder)
    return embedder, db, agent


@st.cache_resource
def init_flashback_filter(_pg_conn):
    '''Phase 2: PE-based Flashback gate. Replaces CLIPFilter.'''
    if not FLASHBACK_ENABLED:
        return None
    fb = FlashbackFilter(
        top_k=FLASHBACK_TOP_K,
        encoder_model=FLASHBACK_ENCODER,
    )
    fb.load(pg_conn=_pg_conn)
    return fb



@st.cache_resource
def get_perception_engine(model_key: str) -> PerceptionEngine:
    """
    Loads and caches the VLM for the given model_key.
    Switching model_key creates a new cached instance without
    unloading the previous one (decoupled by design).
    """
    engine = PerceptionEngine(model_key)
    engine.load()
    return engine


def severity_icon(sev):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚪")


def severity_badge_class(sev):
    if sev == "HIGH":
        return "av-status-alert"
    if sev == "MEDIUM":
        return "av-status-watch"
    return ""


def extract_summary_for_context(caption: str) -> str:
    """
    Pull Summary text to carry forward to next chunk.
    Only Summary should be propagated as temporal context.
    """
    if not caption:
        return ""
    m = re.search(
        r"Summary:\s*(.+?)(?:\n\s*EVENT:|$)",
        caption,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return " ".join(m.group(1).strip().split())
    # If Summary is missing, do not pass arbitrary text as rolling context.
    return ""


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("# AgentVigil")
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

    # ── Model selection ────────────────────────────────────────────
    st.sidebar.markdown("**🤖 Model Selection**")
    model_options = list(VLM_REGISTRY.keys())
    default_model_idx = model_options.index(DEFAULT_VLM) if DEFAULT_VLM in model_options else 0
    selected_model = st.sidebar.selectbox(
        "VLM Model",
        model_options,
        index=default_model_idx,
        format_func=lambda k: VLM_REGISTRY[k]["display_name"],
        help="Select the vision-language model for video captioning.",
    )
    model_cfg = VLM_REGISTRY[selected_model]
    st.sidebar.caption(f"_{model_cfg['description']}_")

    st.sidebar.divider()

    # ── Prompt selection ─────────────────────────────────────────────
    st.sidebar.markdown("**📝 Prompt**")
    prompt_registry = get_prompt_registry()
    default_stem = get_default_stem()
    if prompt_registry:
        stem_list = list(prompt_registry.keys())
        default_idx = stem_list.index(default_stem) if default_stem in stem_list else 0
        selected_stem = st.sidebar.selectbox(
            "Prompt file",
            stem_list,
            index=default_idx,
            format_func=lambda s: prompt_registry[s]["name"],
            help="Select a prompt template from the prompts/ folder.",
        )
    else:
        selected_stem = default_stem

    return page


# ═════════════════════════════════════════════════════════════════════
# PAGE: PROCESS VIDEO
# ═════════════════════════════════════════════════════════════════════

def page_process(embedder, db, agent, flashback_filter):
    # Read current selections from session state
    model_key = st.session_state.get("selected_model", DEFAULT_VLM)
    prompt_stem = st.session_state.get("prompt_stem", get_default_stem())
    model_cfg = VLM_REGISTRY[model_key]

    safe_model = html.escape(model_cfg["display_name"])
    st.markdown(
        f'<div class="av-page-card">'
        f'<h1>Process Surveillance Video</h1>'
        f'<div class="av-subtitle">Model: <strong>{safe_model}</strong> &nbsp;·&nbsp; '
        f'Pipeline: Adaptive Chunk → CLIP Filter → VLM Caption → Agent</div>',
        unsafe_allow_html=True,
    )

    cam_label_col, reset_label_col = st.columns([3, 1])
    with cam_label_col:
        st.markdown("**Camera ID**")
    with reset_label_col:
        st.markdown("**Actions**")

    cam_col, reset_col = st.columns([3, 1])
    with cam_col:
        existing_cameras = db.get_cameras()
        camera_options = sorted(set(["CAM-01", "CAM-02", "CAM-03", "CAM-04"] + existing_cameras))
        camera_id = st.selectbox(
            "Camera ID",
            options=camera_options + ["＋ New camera ID..."],
            index=0,
            label_visibility="collapsed",
            help="Tag this video to a camera. Tier 2 semantic search boosts matches from the same camera.",
        )
        if camera_id == "＋ New camera ID...":
            camera_id = st.text_input("Enter new camera ID", placeholder="e.g. CAM-05 / Entrance-North")
            if not camera_id:
                camera_id = "CAM-01"

    with reset_col:
        with st.popover("🗑️ Reset DB", width="stretch"):
            st.warning("This will delete stored captions, scores and vectors. Keywords and CLIP prompts are kept.")
            scope = st.radio("Scope", ["This camera only", "All cameras"], index=0)
            if st.button("Confirm reset", type="primary", width="stretch"):
                cam_filter = camera_id if scope == "This camera only" else None
                result = db.reset_data(camera_id=cam_filter)
                st.success(
                    f"Reset complete — {result['pg_chunks_deleted']} chunks, "
                    f"{result['pg_videos_deleted']} videos deleted ({result['scope']})."
                )
                st.cache_resource.clear()
                st.rerun()

    st.session_state["camera_id"] = camera_id

    uploaded = st.file_uploader(
        "Upload video", type=["mp4", "avi", "mov", "mkv"]
    )
    if not uploaded:
        st.info("Upload a surveillance video to begin.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown('</div>', unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    video_name = os.path.splitext(uploaded.name)[0]
    info = get_video_info(tmp_path)

    # ── Compact video + info row ────────────────────────────────────
    with st.expander("🎬 Preview & Video Info", expanded=True):
        vcol, icol = st.columns([1, 2])
        with vcol:
            st.video(tmp_path)
        with icol:
            st.caption("**Video metadata**")
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Duration", f"{info['duration_sec']}s")
            r1c2.metric("FPS", f"{info['fps']}")
            r1c3.metric("Resolution", f"{info['width']}×{info['height']}")
            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("Chunk Size", f"{info['chunk_duration']}s")
            r2c2.metric("Stride", f"{info['stride_sec']}s")
            r2c3.metric("Est. Chunks", f"{info['total_chunks']}")

    if not st.button(
        "🚀 Run AgentVigil Pipeline",
        type="primary",
        width="stretch",
    ):
        return

    # Load (or retrieve from cache) the selected perception engine
    with st.spinner(f"Loading {model_cfg['display_name']}..."):
        perception = get_perception_engine(model_key)

    try:
        run_pipeline(
            tmp_path, video_name, info,
            perception, embedder, db, agent, flashback_filter,
            prompt_stem=prompt_stem,
            camera_id=camera_id,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_pipeline(
    video_path, video_name, info,
    perception, embedder, db, agent, flashback_filter,
    prompt_stem: str = "standard",
    camera_id: str = "CAM-01",
):
    pipeline_start = time.time()

    # ── STAGE 0: Adaptive Chunking ──────────────────────────────────
    with st.status(
        "⏳ Stage 0: Adaptive Chunking & Frame Extraction...",
        expanded=False,
    ) as s0:
        output_dir = os.path.join(
            os.getenv("AGENTVIGIL_TEMP_DIR", tempfile.gettempdir()),
            "agentvigil",
            f"temp_workspace_{video_name}",
        )
        os.makedirs(output_dir, exist_ok=True)
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

    with st.status(
        "⏳ Flashback Filter: PE retrieval scoring...", expanded=False
    ) as s_fb:
        if flashback_filter is None:
            # Flashback disabled in config — pass everything through
            passed_chunks  = chunks
            skipped_chunks = {}
            fb_stats = {"enabled": False, "total": len(chunks),
                        "passed": len(chunks), "skipped": 0,
                        "filter_rate": 0.0, "compute_saved_pct": 0.0,
                        "scores": {}}
            s_fb.update(label="⚪ Flashback: Disabled — all chunks pass",
                        state="complete")
        else:
            passed_chunks, skipped_chunks, fb_stats = (
                flashback_filter.filter_chunks(
                    chunks, threshold=FLASHBACK_THRESHOLD,
                )
            )
            mem_stats = fb_stats.get("memory_stats", {})
            s_fb.update(
                label=(
                    f"✅ Flashback: {fb_stats['passed']}/{fb_stats['total']} passed "
                    f"({fb_stats['compute_saved_pct']}% filtered) | "
                    f"K={fb_stats.get('top_k','?')} | "
                    f"{mem_stats.get('normal_count','?')}n / "
                    f"{mem_stats.get('anomalous_count','?')}a memory"
                ),
                state="complete",
            )
            st.session_state["_flashback_stats"] = fb_stats

    if fb_stats.get("enabled") and fb_stats.get("scores"):
        st.markdown("#### Flashback Anomaly Scores")
        scores = fb_stats["scores"]
        score_df = pd.DataFrame([
            {
                "Chunk":  f"C{k}",
                "Flashback Score": v,
                "Status": "✅ VLM" if k in passed_chunks else "❌ Skip",
            }
            for k, v in sorted(scores.items())
        ])
        st.bar_chart(score_df.set_index("Chunk")["Flashback Score"])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("→ VLM",       fb_stats["passed"])
        mc2.metric("Skipped",     fb_stats["skipped"])
        mc3.metric("Compute Saved", f"{fb_stats['compute_saved_pct']}%")

    # ── Process SKIPPED chunks (auto Normal, no VLM) ────────────────
    for cidx, cinfo in sorted(skipped_chunks.items()):
        retrieved = cinfo.get("flashback_top_captions", [])[:1]
        retrieved_str = retrieved[0] if retrieved else "Normal activity."

        embedding = embedder.embed_text(retrieved_str)
        db.store_chunk(
            video_name=video_name,
            chunk_index=cidx,
            start_sec=cinfo["start_sec"],
            end_sec=cinfo["end_sec"],
            num_frames=cinfo["num_frames"],
            caption=(
                f"[Flashback filtered — score: "
                f"{cinfo.get('flashback_score', 0):.3f}] {retrieved_str}"
            ),
            event_type="Normal Activity",
            embedding=embedding,
            anomaly_score=0.0,
            camera_id=camera_id,
            metadata={
                "flashback_score":      cinfo.get("flashback_score", 0),
                "flashback_filtered":   True,
                "flashback_top_caps":   cinfo.get("flashback_top_captions", []),
                "flashback_top_cats":   cinfo.get("flashback_top_categories", []),
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
    rolling_context = ""

    for i, (cidx, cinfo) in enumerate(sorted(passed_chunks.items())):
        progress.progress(
            (i + 1) / max(total_to_process, 1),
            text=(
                f"Chunk {cidx} "
                f"({cinfo['start_sec']:.0f}s-{cinfo['end_sec']:.0f}s) | "
                f"{i+1}/{total_to_process}"
            ),
        )

        # Build Flashback prior — top retrieved captions as scene priors
        fb_prior = ""
        if FLASHBACK_FEED_CAPTIONS_TO_VLM:
            top_caps = cinfo.get("flashback_top_captions", [])
            top_cats = cinfo.get("flashback_top_categories", [])
            K = min(FLASHBACK_VLM_PRIOR_K, len(top_caps))
            if K > 0:
                lines = [
                    f"  {i+1}. [{top_cats[i] if i < len(top_cats) else '?'}] "
                    f"{top_caps[i]}"
                    for i in range(K)
                ]
                fb_prior = (
                    "Retrieved scene priors (similarity-ranked from memory):\\n"
                    + "\\n".join(lines)
                )

                # Stage D: category shortlist — constrain the VLM to the
                # gate's retrieved categories to kill the "magnet" effect
                # (RoadAccidents/Assault/Vandalism absorbing everything).
                # Matches benchmark.py Stage D so the live app and the
                # evaluated pipeline classify identically.
                shortlist = []
                for c in top_cats[:K]:
                    if c and c not in shortlist and c != "Normal_Videos_event":
                        shortlist.append(c)
                if shortlist:
                    fb_prior += (
                        "\\n\\nMOST LIKELY CATEGORIES for this scene (from retrieval): "
                        + ", ".join(shortlist)
                        + ".\\nStrongly prefer one of these categories if the footage "
                        "supports it. Only choose a different category if the footage "
                        "clearly shows something else. Use Normal_Videos_event only if "
                        "no anomaly is visible."
                    )

        caption, latency = perception.caption_chunk(
            cinfo["frame_paths"],
            prompt_type=prompt_stem,
            prev_context=rolling_context if CHUNK_CONTEXT_ENABLED else "",
            flashback_prior=fb_prior,
        )
        event_type = perception.extract_event_type(caption)
        embedding = embedder.embed_text(caption)

        if CHUNK_CONTEXT_ENABLED:
            next_ctx = extract_summary_for_context(caption)
            rolling_context = next_ctx[:CHUNK_CONTEXT_MAX_CHARS]

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
            camera_id=camera_id,
            metadata={
            "flashback_score":      cinfo.get("flashback_score", 0),
            "flashback_filtered":   False,
            "flashback_top_caps":   cinfo.get("flashback_top_captions", []),
            "flashback_top_cats":   cinfo.get("flashback_top_categories", []),
            "latency":              latency,
            "model":                perception.model_key,
            "prompt_stem":          prompt_stem,
        },
        )

        # Stage 2: Adaptive Decision Router
        analysis = agent.analyze_chunk(
            video_name=video_name,
            chunk_index=cidx,
            caption=caption,
            embedding=embedding,
            pg_id=pg_id,
            camera_id=camera_id,
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

        sev = severity_icon(analysis["severity"])
        with results_container.expander(
            f"{sev} Chunk {cidx} | "
            f"{cinfo['start_sec']:.0f}-{cinfo['end_sec']:.0f}s | "
            f"{event_type} | Tier {analysis['tier_used']} | "
            f"Score: {analysis['anomaly_score']} | {latency:.1f}s",
            expanded=False,
        ):
            # ── Top row: metrics + frame viewer button ───────────────
            mc1, mc2, mc3, mc4, btn_col = st.columns([1, 1, 1, 1, 1.2])
            mc1.metric("Anomaly Score", f"{analysis['anomaly_score']}")
            mc2.metric("Tier Used", f"{analysis['tier_used']}")
            mc3.metric("Similar Past", f"{analysis['similar_count']}")
            mc4.metric("CLIP Score", f"{cinfo.get('clip_score', 'N/A')}")

            frame_paths = cinfo.get("frame_paths", [])
            with btn_col:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if frame_paths:
                    with st.popover(
                        f"🖼️ View Frames ({len(frame_paths)})",
                        width="stretch",
                    ):
                        st.caption(
                            f"**Chunk {cidx}** · "
                            f"{cinfo['start_sec']:.0f}s – {cinfo['end_sec']:.0f}s · "
                            f"{len(frame_paths)} frames extracted"
                        )
                        gallery_cols = 5
                        for start in range(0, len(frame_paths), gallery_cols):
                            row_paths = frame_paths[start : start + gallery_cols]
                            row_cols = st.columns(len(row_paths))
                            for k, fp in enumerate(row_paths):
                                row_cols[k].image(
                                    fp,
                                    caption=f"f{start + k}",
                                    width="stretch",
                                )

            # ── Caption ─────────────────────────────────────────────
            st.markdown(f"**Caption:** {caption}")

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
            video_name, info["duration_sec"], camera_id=camera_id
        )
        s3.update(
            label="✅ Stage 3: Analysis complete", state="complete"
        )

    pipeline_time = time.time() - pipeline_start

    # ── RESULTS DASHBOARD ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Analysis Results")

    overall_severity = video_analysis.get("overall_severity", "LOW")
    overall_alert = video_analysis.get("overall_alert", "NORMAL")
    sev_i = severity_icon(overall_severity)
    alert_text = "YES" if overall_alert == "ALERT" else "NO"

    safe_primary_event = html.escape(str(video_analysis.get("primary_event", "Unknown")))
    st.markdown(
        f"""
        <div class="av-status-grid">
            <div class="av-status-card {severity_badge_class(overall_severity)}">
                <div class="av-status-label">Primary Event</div>
                <div class="av-status-value">{safe_primary_event}</div>
            </div>
            <div class="av-status-card">
                <div class="av-status-label">Peak Score</div>
                <div class="av-status-value">{video_analysis.get('overall_score', 0):.3f}</div>
            </div>
            <div class="av-status-card {severity_badge_class(overall_severity)}">
                <div class="av-status-label">Severity</div>
                <div class="av-status-value">{sev_i} {overall_severity}</div>
            </div>
            <div class="av-status-card {severity_badge_class(overall_severity)}">
                <div class="av-status-label">Critical Flag</div>
                <div class="av-status-value">{alert_text}</div>
            </div>
            <div class="av-status-card">
                <div class="av-status-label">Anomalous Chunks</div>
                <div class="av-status-value">{video_analysis.get('anomalous_chunks', 0)}/{video_analysis.get('total_chunks', 0)}</div>
            </div>
            <div class="av-status-card">
                <div class="av-status-label">Pipeline Time</div>
                <div class="av-status-value">{pipeline_time:.1f}s</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔀 Adaptive Router — Tier Usage")
    t0, t1, t2, t3 = st.columns(4)
    t0.metric("T0 CLIP Skip", tier_counts.get(0, 0))
    t1.metric("T1 Keyword", tier_counts.get(1, 0))
    t2.metric("T2 Semantic", tier_counts.get(2, 0))
    t3.metric("T3 Temporal", tier_counts.get(3, 0))

    st.markdown("### 📝 Agent Narrative")
    st.info(video_analysis.get("narrative", "No narrative."))

    if video_analysis.get("chunk_details"):
        st.markdown("### 📋 Chunk Timeline")
        df = pd.DataFrame(video_analysis["chunk_details"])
        st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("### ⚡ Pipeline Performance")
    if chunk_results:
        latencies = [r["latency"] for r in chunk_results]
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Avg VLM Latency", f"{sum(latencies)/len(latencies):.1f}s")
        p2.metric("Total VLM Calls", f"{len(chunk_results)}")
        p3.metric("Chunks Skipped (CLIP)", f"{len(skipped_chunks)}")
        p4.metric(
            "VLM Compute Saved",
            f"{len(skipped_chunks) * (sum(latencies)/max(len(latencies), 1)):.0f}s",
        )

    # Clean up extracted frame JPEGs for this video.
    shutil.rmtree(output_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ═════════════════════════════════════════════════════════════════════

def page_alerts(db):
    st.title("Alert Dashboard")

    cameras = db.get_cameras()
    cam_filter = None
    if cameras:
        options = ["All cameras"] + cameras
        sel = st.selectbox("Filter by camera", options, index=0)
        cam_filter = None if sel == "All cameras" else sel

    alerts = db.get_alerts(limit=50, camera_id=cam_filter)
    if not alerts:
        st.info("No alerts yet. Process a video to generate alerts.")
        return

    alert_count = sum(1 for a in alerts if a["alert_status"] == "ALERT")
    watch_count = sum(1 for a in alerts if a["alert_status"] == "WATCH")
    ac1, ac2 = st.columns(2)
    ac1.metric("🔴 Active Alerts", alert_count)
    ac2.metric("🟡 Watch Items", watch_count)

    st.markdown("---")

    for a in alerts:
        sev = severity_icon(a.get("severity", "LOW"))
        cam_label = a.get("camera_id", "?")
        with st.expander(
            f"{sev} [{cam_label}] {a['event_type']} | {a['video_name']} "
            f"C{a['chunk_index']} | Score: {a['anomaly_score']} | "
            f"Tier {a.get('tier_used', '?')}",
            expanded=(a["alert_status"] == "ALERT"),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", f"{a.get('anomaly_score', 0):.2f}")
            c2.metric("Time", f"{a['start_sec']}s-{a['end_sec']}s")
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
    st.title("Search Similar Incidents")
    st.caption("Describe a scenario — Milvus finds similar past incidents via vector similarity search.")

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
                    st.metric("Past Anomaly Score", f"{r.get('anomaly_score', 0):.2f}")
        else:
            st.warning("No similar incidents found.")


# ═════════════════════════════════════════════════════════════════════
# PAGE: STATS
# ═════════════════════════════════════════════════════════════════════

def page_stats(db):
    st.title("Statistics")

    cameras = db.get_cameras()
    cam_filter = None
    if cameras:
        options = ["All cameras"] + cameras
        sel = st.selectbox("Filter by camera", options, index=0)
        cam_filter = None if sel == "All cameras" else sel

    stats = db.get_event_stats(camera_id=cam_filter)
    if not stats:
        st.info("No data yet. Process videos to see statistics.")
        return

    df = pd.DataFrame(stats)
    st.markdown("### Event Distribution")
    st.bar_chart(df.set_index("event_type")["count"])
    st.markdown("### Details")
    st.dataframe(df, width="stretch", hide_index=True)


# ═════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ═════════════════════════════════════════════════════════════════════

def page_history(db):
    st.title("Video History")

    cameras = db.get_cameras()
    cam_filter = None
    if cameras:
        options = ["All cameras"] + cameras
        sel = st.selectbox("Filter by camera", options, index=0)
        cam_filter = None if sel == "All cameras" else sel

    videos = db.get_all_videos(camera_id=cam_filter)
    if not videos:
        st.info("No videos processed yet.")
        return

    for v in videos:
        sev = severity_icon(v.get("overall_severity", "LOW"))
        cam_label = v.get("camera_id", "?")
        with st.expander(
            f"{sev} [{cam_label}] {v['video_name']} | {v['primary_event']} | "
            f"Score: {v.get('overall_score', 0):.2f} | "
            f"{v.get('anomalous_chunks', 0)}/"
            f"{v.get('total_chunks', 0)} anomalous"
        ):

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric("Duration", f"{v.get('duration_sec', 0):.1f}s")
            c2.metric("Chunks", v.get('total_chunks', 0))
            c3.metric("Score", f"{v.get('overall_score', 0):.2f}")

            # Parse timestamp
            ts = str(v.get("processed_at", ""))
            date_display = "—"
            time_display = ""
            try:
                dt = datetime.fromisoformat(ts[:19])
                date_display = dt.strftime("%b %d")   # "Apr 18"
                time_display = dt.strftime("%I:%M %p")  # "05:48 AM"
            except Exception:
                date_display = ts[:10] if len(ts) >= 10 else ts

            c4.metric("Date", date_display)
            c5.metric("Time", time_display)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    try:
        embedder, db, agent = init_core_system()
        flashback_filter = init_flashback_filter(db.pg_conn)  # leading _ tells Streamlit not to hash pg_conn
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
        page_process(embedder, db, agent, flashback_filter)
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
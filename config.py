"""
AgentVigil — Central Configuration
====================================
Chunking tiers and strategies replicated EXACTLY from benchmark_vlm.py
which produced our winning results across 4 crime categories.

BENCHMARK-PROVEN VALUES — DO NOT CHANGE WITHOUT RE-BENCHMARKING:
  - Chunk tiers: (30,3) (120,4) (300,6) (1800,10) (inf,15)
  - Sampling: "uniform" strategy
  - Frames: 1fps extraction
  - Max chunks: 50 (benchmark) / 100 (production)

PROMPTS
  Prompt text does NOT live here.
  See the prompts/ folder — one .py file per variant.
  Use prompt_loader.get_prompt_registry() to enumerate them at runtime.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── VLM Registry ───────────────────────────────────────────────────
VLM_REGISTRY = {
    "LLaVA-OneVision-7B": {
        "adapter": "LLaVAAdapter",
        "model_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "display_name": "LLaVA-OneVision-7B (4-bit)",
        "description": "Production model. 7B params, image-based, 4-bit quant.",
    },
    "VideoLLaMA3-2B": {
        "adapter": "VideoLLaMA3Adapter",
        "model_id": "DAMO-NLP-SG/VideoLLaMA3-2B",
        "display_name": "VideoLLaMA3-2B",
        "description": "Lightweight video-native model. Lower VRAM, faster.",
    },
    "Qwen2.5-VL-3B": {
    "adapter": "Qwen25VLAdapter",
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "display_name": "Qwen2.5-VL-3B-Instruct (4-bit)",
    "description": (
        "Video-native VLM with dynamic FPS sampling. ~10s/chunk "
        "latency at 4-bit on 12GB VRAM. Use when categorical "
        "metrics are the bottleneck."
    ),
},
}

DEFAULT_VLM = "VideoLLaMA3-2B"

# Legacy aliases
PRIMARY_VLM = VLM_REGISTRY[DEFAULT_VLM]["model_id"]
PRIMARY_VLM_NAME = VLM_REGISTRY[DEFAULT_VLM]["display_name"]

# ── Prompt folder ──────────────────────────────────────────────────
PROMPTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# ── Embedding Model ─────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Adaptive Chunking ───────────────────────────────────────────────
CHUNK_DURATION_TIERS = [
    (30,           10),
    (120,          20),
    (300,          30),
    (1800,         50),
    (float('inf'), 60),
]

MAX_CHUNKS = 100
FRAMES_PER_SECOND = 2
CHUNK_OVERLAP_SEC = 0.0
CHUNK_CONTEXT_ENABLED = True
CHUNK_CONTEXT_MAX_CHARS = 320
# Sampling strategy when total chunks exceed MAX_CHUNKS
# Options: "uniform" (default, benchmark-proven), "edges", "all"
SAMPLING_STRATEGY = "uniform"

# ── Sliding Window (OPTIONAL — benchmark did NOT use this) ──────────
# Set to 0.0 to replicate exact benchmark behavior (hard cuts).
# Set to 1.0 for 1-second overlap between chunks (catches boundary events).
# Recommendation: 0.0 for benchmarking, 1.0 for production deployment.

# ── Flashback (Phase 2) — replaces CLIP pre-filter ───────────────────
FLASHBACK_ENABLED               = True
FLASHBACK_ENCODER               = "PE-Core-L14-336"  # PE backbone
FLASHBACK_TOP_K                 = 10                  # paper's K = 10
FLASHBACK_THRESHOLD             = 0.25                # gate threshold; same scale as old CLIP_ANOMALY_THRESHOLD
FLASHBACK_ALPHA                 = 0.95                # SAP — applied at memory-build time, not here
FLASHBACK_FRAMES_PER_SEGMENT    = 16                  # paper's Tsample
FLASHBACK_INPUT_RES             = 448

# When Flashback gate passes a chunk, also pass its top retrieved
# captions to the VLM as a prior context block. Set False to disable
# the hybrid mode and run the gate alone.
FLASHBACK_FEED_CAPTIONS_TO_VLM  = True

# Number of retrieved captions injected into the VLM prompt (≤ FLASHBACK_TOP_K).
# Keep small to avoid drowning the VLM in priors.
FLASHBACK_VLM_PRIOR_K           = 3

# ── Databases ───────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "agentvigil")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = "agentvigil_captions"

# ── Adaptive Decision Router Thresholds ─────────────────────────────
TIER1_THRESHOLD = 0.3
TIER2_THRESHOLD = 0.6
ALERT_THRESHOLD = 0.7
WATCH_THRESHOLD = 0.4
SIMILAR_INCIDENT_TOP_K = 5

# ── Event Categories ────────────────────────────────────────────────
EVENT_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
    "Normal_Videos_event",
]

# ── Legacy prompt aliases ──────────────────────────────────────────
# Any file that does `from config import SYSTEM_PROMPT` will get the
# default prompt loaded lazily from the prompts/ folder.
def __getattr__(name):
    if name in ("SYSTEM_PROMPT", "FEW_SHOT_EXAMPLES"):
        from prompt_loader import get_default_stem, load_prompt
        p = load_prompt(get_default_stem())
        return p["system"] if name == "SYSTEM_PROMPT" else p["few_shot"]
    raise AttributeError(f"module 'config' has no attribute '{name}'")
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
FRAMES_PER_SECOND = 10
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

# ── CLIP Pre-Filter ─────────────────────────────────────────────────
CLIP_ENABLED = True
CLIP_MODEL = "ViT-B/32"
CLIP_ANOMALY_THRESHOLD = 0.25

CLIP_ANOMALY_PROMPTS = [
    "a car accident on the road",
    "a vehicle collision with debris",
    "a person fighting another person",
    "a violent assault in progress",
    "a robbery or mugging",
    "a person stealing from a store",
    "a person vandalizing property",
    "a fire burning in a building",
    "a person running away from a crime scene",
    "a person with a weapon",
    "suspicious activity at night",
    "a break-in or burglary",
]

CLIP_NORMAL_PROMPTS = [
    "people walking normally on a street",
    "cars driving on a road following traffic rules",
    "an empty quiet parking lot",
    "normal traffic at an intersection",
    "a quiet store with customers shopping peacefully",
    "people sitting on a bench",
    "a calm residential neighborhood",
]

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
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
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── VLM (Stage 1) ──────────────────────────────────────────────────
PRIMARY_VLM = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
PRIMARY_VLM_NAME = "LLaVA-OneVision-7B"

# ── Embedding Model ─────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── Adaptive Chunking (EXACT copy from benchmark_vlm.py) ───────────
# These produced our best results. Do not change without re-benchmarking.
CHUNK_DURATION_TIERS = [
    (30,           3),     # Videos ≤ 30s   → 3s chunks
    (120,          4),     # Videos ≤ 2min  → 4s chunks (benchmark sweet spot)
    (300,          6),     # Videos ≤ 5min  → 6s chunks
    (1800,         10),    # Videos ≤ 30min → 10s chunks
    (float('inf'), 15),   # Videos > 30min → 15s chunks
]

MAX_CHUNKS = 100                    # Production cap (benchmark used 50)
FRAMES_PER_SECOND = 1               # 1 frame per second within each chunk

# Sampling strategy when total chunks exceed MAX_CHUNKS
# Options: "uniform" (default, benchmark-proven), "edges", "all"
SAMPLING_STRATEGY = "uniform"

# ── Sliding Window (OPTIONAL — benchmark did NOT use this) ──────────
# Set to 0.0 to replicate exact benchmark behavior (hard cuts).
# Set to 1.0 for 1-second overlap between chunks (catches boundary events).
# Recommendation: 0.0 for benchmarking, 1.0 for production deployment.
CHUNK_OVERLAP_SEC = 0.0

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
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
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
    "Road Accident / Vehicle Collision",
    "Robbery / Armed Robbery",
    "Burglary / Breaking and Entering",
    "Fighting / Assault",
    "Shoplifting / Stealing",
    "Vandalism / Property Damage",
    "Arson / Fire",
    "Normal Activity",
]

# ── Surveillance Keywords (Tier 1) ─────────────────────────────────
EVENT_KEYWORDS = {
    "Road Accident / Vehicle Collision": [
        "accident", "collision", "crash", "impact", "hit", "rear-end",
        "wreck", "smash", "vehicle", "debris", "rollover",
    ],
    "Robbery / Armed Robbery": [
        "robbery", "rob", "robbing", "armed", "holdup", "mugging",
        "threaten", "weapon", "gun", "knife", "demand",
    ],
    "Fighting / Assault": [
        "fight", "fighting", "assault", "punch", "kick", "attack",
        "brawl", "violence", "struggle", "hit", "beating",
    ],
    "Shoplifting / Stealing": [
        "steal", "shoplifting", "theft", "shoplift", "grab", "conceal",
        "pocket", "snatch", "take", "merchandise",
    ],
    "Vandalism / Property Damage": [
        "vandalism", "vandal", "damage", "smash", "destroy", "graffiti",
        "break", "shatter", "deface", "litter",
    ],
    "Arson / Fire": [
        "arson", "fire", "flames", "burning", "ignite", "smoke",
        "explosion", "torch", "blaze",
    ],
}

ALL_CRIME_KEYWORDS = list(set(
    kw for keywords in EVENT_KEYWORDS.values() for kw in keywords
))

# ── Surveillance Prompt (same as benchmark_vlm.py) ──────────────────
SYSTEM_PROMPT = """You are an expert AI surveillance analyst for the AgentVigil security system.
Your task is to analyze a sequence of video frames extracted from CCTV/surveillance footage and
produce a precise, factual description of the event occurring.

Focus on:
1. **Actions and movements**: What are the subjects doing? Are they running, colliding, grabbing, breaking, fighting?
2. **Anomalous behavior**: Identify anything that deviates from normal activity — sudden motion, impact, aggression, stealth.
3. **Temporal progression**: Describe how the scene evolves across the frames (e.g., "In the first frames... then... finally...").
4. **Key objects**: Vehicles, weapons, tools, bags, broken items — anything relevant to the event.

You MUST classify the event into one of these categories if applicable:
- Road Accident / Vehicle Collision
- Robbery / Armed Robbery
- Burglary / Breaking and Entering
- Fighting / Assault
- Shoplifting / Stealing
- Vandalism / Property Damage
- Arson / Fire
- Normal Activity (only if nothing anomalous is detected)

Be specific and decisive. Do NOT give vague descriptions like "a busy street with cars"."""

FEW_SHOT_EXAMPLES = """Here are examples of the analysis quality expected:

Example 1 — Road Accident:
"A white sedan traveling at high speed rear-ends a stationary black SUV at an intersection. The first frames show the sedan approaching rapidly. In the middle frames, the moment of collision is visible with debris scattering. The final frames show both vehicles stopped with the sedan's hood crumpled. EVENT: Road Accident / Vehicle Collision."

Example 2 — Robbery:
"Two individuals approach a person walking on a sidewalk from behind. One grabs the victim's bag while the other blocks their path. The victim struggles briefly before the assailants run away with the bag toward a side alley. EVENT: Robbery."

Example 3 — Fighting:
"Two men face each other aggressively in a parking lot. In the initial frames, one pushes the other. The confrontation escalates as both begin throwing punches. Bystanders start moving away. EVENT: Fighting / Assault."

Example 4 — Shoplifting:
"A person in a store aisle looks around, then quickly places merchandise into their jacket pocket. They then walk briskly toward the exit without approaching the checkout counter. EVENT: Shoplifting / Stealing."

Example 5 — Vandalism:
"An individual approaches a parked car and uses a blunt object to smash the side window. Glass fragments scatter. The person then moves to the next vehicle. EVENT: Vandalism / Property Damage."

Example 6 — Normal Activity:
"Pedestrians walk along a sidewalk at a steady pace. Vehicles move through the intersection following traffic signals. No unusual or anomalous behavior is detected. EVENT: Normal Activity."

Now analyze the following surveillance frames:"""
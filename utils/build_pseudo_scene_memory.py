"""
Build the Pseudo-Scene Memory using the Anthropic API
======================================================
ONE-TIME OFFLINE COST. Run this once, before the first video is processed.
Re-run only when you want to extend the memory.

WHAT THIS DOES
--------------
For each iteration we ask Claude to invent N normal/anomalous scene
description PAIRS following the Flashback paper's JSON schema:

    {"descriptions": [
        {
          "normal":    {"category": "...", "description": "..."},
          "anomalous": {"category": "...", "description": "..."}
        },
        ...
    ]}

Each pair becomes 2 MemoryEntry rows (one normal, one anomalous), so
NUM_PAIRS pairs = 2 * NUM_PAIRS rows in the database.

WHY PAIRS, NOT SEPARATE LISTS
-----------------------------
The paper's prompt asks for paired examples specifically — pairs anchor
each anomalous description to a thematically-matched normal one
("Robbery: armed individual demands money" ↔ "Cashier: customer pays
quietly"). This pairing is what RP later widens in feature space.

USAGE
-----
    # Set your API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # Build a small starter memory (~1k pairs, ~10 minutes)
    python scripts/build_pseudo_scene_memory.py --pairs 1000

    # Build the full paper-scale memory (~1M pairs, ~76 hours, $$$)
    python scripts/build_pseudo_scene_memory.py --pairs 1000000

    # Resume a partial build
    python scripts/build_pseudo_scene_memory.py --pairs 1000 --resume

COST GUIDE (rough, Claude Sonnet 4.x pricing)
----------------------------------------------
    1k pairs   ≈   $5
    10k pairs  ≈  $50
    100k pairs ≈ $300
    1M pairs   ≈ usd low-thousands  (paper paid $181 with GPT-4o)

We default to 1000 pairs — enough for a working demo. Scale up later.

PIPELINE
--------
    1. Anthropic API → JSON of pairs
    2. parse + flatten into MemoryEntry rows
    3. apply_rp() per row
    4. PE encode_texts() in batches
    5. Anomalous embeddings × α (SAP)
    6. INSERT into pseudo_scene_memory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterable, List

import psycopg2

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import POSTGRES_URL                                # noqa: E402
from perception_encoder import PerceptionEncoder               # noqa: E402
from pseudo_scene_memory import PseudoSceneMemory, MemoryEntry # noqa: E402


# ── Prompt design (paper §3.2 Context + Format prompts) ─────────────
# We keep the wording close to the paper but make the JSON structure
# explicit so Claude returns a parseable response.

CONTEXT_PROMPT = """You are helping build a video anomaly detection knowledge base.

For surveillance video, anomalous events are rare but their categories are diverse: \
robbery, shoplifting, road accidents, vandalism, fighting, arson, burglary, abuse, \
assault, explosion, shooting, stealing, arrest, and so on.

Generate {n} pairs of scene descriptions. Each pair contains:
  - one NORMAL scene (a routine, peaceful, or non-criminal activity that could plausibly \
appear in surveillance footage)
  - one ANOMALOUS scene (a clear criminal, dangerous, or rule-violating activity)

The two halves of each pair should describe THEMATICALLY-RELATED but \
behaviorally-OPPOSITE situations — e.g., paired with "Cashier processing payment" \
should be "Armed person demanding money from cashier", not "House on fire".

Each description must be:
  - 1 sentence, 10-25 words
  - concrete and visual (people, actions, objects)
  - free of judgmental language ("evil", "bad guy") — describe the ACTION, not opinion
  - varied across location, time-of-day, and people

Vary the categories. Across the {n} pairs, cover at least 12 distinct anomalous \
categories and 12 distinct normal categories.

Return ONLY valid JSON, no commentary, exactly this schema:
{{
  "descriptions": [
    {{
      "normal":    {{"category": "...", "description": "..."}},
      "anomalous": {{"category": "...", "description": "..."}}
    }}
    // ... repeat {n} times total
  ]
}}"""


def call_claude(n_pairs: int, model: str, max_tokens: int) -> List[dict]:
    """
    One API call → up to n_pairs pairs of {normal, anomalous} dicts.

    We chunk the build into batches of ~50 pairs per call. Larger
    batches risk JSON truncation; smaller batches waste tokens on the
    prompt boilerplate.
    """
    from anthropic import Anthropic

    client = Anthropic()  # picks up ANTHROPIC_API_KEY from env
    prompt = CONTEXT_PROMPT.format(n=n_pairs)

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        b.text for b in resp.content if getattr(b, "type", None) == "text"
    )

    # Strip code fences if Claude wraps the JSON
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                  flags=re.MULTILINE)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try extracting the largest {...} blob
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise RuntimeError(f"Could not parse JSON: {e}\n--\n{text[:400]}")
        data = json.loads(m.group(0))

    pairs = data.get("descriptions", [])
    if not isinstance(pairs, list):
        raise RuntimeError(f"Unexpected payload shape: {list(data.keys())}")
    return pairs


def pairs_to_entries(pairs: Iterable[dict]) -> List[MemoryEntry]:
    """Flatten {normal, anomalous} pairs into 2 MemoryEntry rows each."""
    entries: List[MemoryEntry] = []
    for p in pairs:
        try:
            n = p["normal"]
            a = p["anomalous"]
            entries.append(MemoryEntry(
                label=0, category=n["category"].strip(),
                text=n["description"].strip(),
            ))
            entries.append(MemoryEntry(
                label=1, category=a["category"].strip(),
                text=a["description"].strip(),
            ))
        except (KeyError, AttributeError, TypeError):
            # Skip malformed pairs rather than abort the whole batch
            continue
    return entries


# ── MAIN ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000,
                    help="Total pairs to generate (= 2 * pairs DB rows).")
    ap.add_argument("--batch", type=int, default=40,
                    help="Pairs per API call. 30-50 is the sweet spot.")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="Anthropic model id.")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="max_tokens per API call.")
    ap.add_argument("--alpha", type=float, default=0.95,
                    help="Scaled Anomaly Penalisation factor (paper default 0.95).")
    ap.add_argument("--encoder", default="PE-Core-G14-448",
                    help="PE backbone (must match the inference encoder).")
    ap.add_argument("--encode-batch", type=int, default=256,
                    help="Texts encoded per PE forward pass.")
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        sys.exit("ANTHROPIC_API_KEY not set — export it before running.")

    # ── Connect to Postgres + ensure schema ─────────────────────────
    print(f"[BUILD] Connecting to {POSTGRES_URL.split('@')[-1]}...")
    pg = psycopg2.connect(POSTGRES_URL)
    pg.autocommit = True
    memory = PseudoSceneMemory(pg)

    # ── Load PE (we encode in big batches at the end of each loop) ─
    print(f"[BUILD] Loading encoder {args.encoder}...")
    pe = PerceptionEncoder(model_name=args.encoder)
    pe.load()

    initial = memory.stats()
    print(f"[BUILD] Memory at start: {initial['total']} rows "
          f"({initial['normal_count']} normal, "
          f"{initial['anomalous_count']} anomalous).")

    # ── Iterate API calls until we hit --pairs ─────────────────────
    total_inserted = 0
    target_rows = 2 * args.pairs       # 2 rows per pair
    start = time.time()

    while total_inserted < target_rows:
        remaining_pairs = (target_rows - total_inserted) // 2
        ask_n = min(args.batch, remaining_pairs)
        if ask_n <= 0:
            break

        try:
            pairs = call_claude(ask_n, args.model, args.max_tokens)
        except Exception as e:
            print(f"[BUILD] API error ({e}) — backing off 30s.")
            time.sleep(30)
            continue

        entries = pairs_to_entries(pairs)
        if not entries:
            print(f"[BUILD] Got 0 valid entries from this call, retrying.")
            continue

        n_inserted = memory.write_pairs(
            entries,
            encoder=pe,
            alpha=args.alpha,
            batch_size=args.encode_batch,
        )
        total_inserted += n_inserted

        elapsed = time.time() - start
        rate = total_inserted / max(elapsed, 1)
        eta  = (target_rows - total_inserted) / max(rate, 0.1)
        print(f"[BUILD] +{n_inserted} rows  "
              f"(total {total_inserted}/{target_rows})  "
              f"{rate:.1f} rows/s  "
              f"ETA {eta/60:.1f} min")

    # ── Final stats ─────────────────────────────────────────────────
    final = memory.stats()
    print("\n[BUILD] Done.")
    print(f"  Normal rows:    {final['normal_count']}")
    print(f"  Anomalous rows: {final['anomalous_count']}")
    print(f"  Categories:     {final['category_count']}")
    print(f"  Wall time:      {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    main()
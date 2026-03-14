"""
VideoLLaMA3 — Full Video Path Test Script
==========================================
Tests VideoLLaMA3-2B using native video_path input (full video mode).
Processor handles all frame sampling internally via ffmpeg/decord.
Includes ground truth loading + BLEU / ROUGE / METEOR / Keyword metrics.

Usage:
    python test_videollama3.py
    python test_videollama3.py --video path/to/video.mp4 --start 10 --end 40
    python test_videollama3.py --fps 2 --max_frames 32 --model DAMO-NLP-SG/VideoLLaMA3-7B

Requirements:
    pip install torch transformers accelerate
    pip install pillow numpy decord ffmpeg-python imageio
    pip install pandas nltk rouge-score
"""

import argparse
import gc
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoProcessor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION — edit these defaults instead of CLI args if preferred
# =============================================================================

VIDEO_INPUT_PATH     = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCF_Crimes\UCF_Crimes\Videos\Robbery\Robbery023_x264.mp4"
UCA_ANNOTATIONS_JSON = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Train.json"
OUTPUT_DIR           = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\benchmark_results"

MODEL_ID             = "DAMO-NLP-SG/VideoLLaMA3-2B"
DEFAULT_FPS          = 1
DEFAULT_MAX_FRAMES   = 128

# =============================================================================
# PROMPT
# =============================================================================

PROMPT = (
    "You are a video analysis assistant. "
    "Describe what is happening in this video in detail. "
    "Focus on: the people present, their actions and movements, "
    "any objects involved, and how the scene changes over time. "
    "Be specific and factual."
)

# =============================================================================
# KEYWORDS  (same set as main benchmark pipeline)
# =============================================================================

EVENT_KEYWORDS = [
    "accident", "collision", "crash", "impact", "hit",
    "robbery", "robbing", "armed", "threaten", "holdup",
    "burglary", "break-in", "breaking", "intrusion",
    "fight", "fighting", "assault", "punch", "attack",
    "steal", "shoplifting", "theft", "shoplift", "grab",
    "vandalism", "damage", "smash", "destroy", "graffiti",
    "arson", "fire", "flames", "burning",
    "suspicious", "anomalous", "unusual", "dangerous", "emergency",
]

# =============================================================================
# GROUND TRUTH LOADER  (same logic as main benchmark pipeline)
# =============================================================================

def load_ground_truth(json_path: str, video_path: str):
    """
    Returns (chunk_gt dict, full_gt string) from UCFCrime annotation JSON.
    chunk_gt maps second -> sentence for granular comparison.
    full_gt is all sentences joined for whole-video metric computation.
    """
    if not os.path.exists(json_path):
        print(f"  [GT] JSON not found: {json_path}")
        return {}, ""

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    short_name = "_".join(video_name.split("_")[:-1])

    with open(json_path, "r") as f:
        annotations = json.load(f)

    info = None
    for key in [video_name, short_name]:
        if key in annotations:
            info = annotations[key]
            break

    if not info:
        print(f"  [GT] No annotation for '{video_name}' or '{short_name}'")
        return {}, ""

    timestamps = info.get("timestamps", [])
    sentences  = info.get("sentences",  [])
    full_gt    = " ".join(sentences)

    # Per-second GT for granular analysis
    chunk_gt = {}
    for (ts, te), sentence in zip(timestamps, sentences):
        for sec in range(int(ts), int(te) + 1):
            chunk_gt[sec] = (chunk_gt.get(sec, "") + " " + sentence).strip()

    print(f"  [GT] {len(sentences)} annotation(s) | "
          f"{len(chunk_gt)} second-level entries")
    return chunk_gt, full_gt


# =============================================================================
# METRICS ENGINE  (identical to main benchmark pipeline)
# =============================================================================

class MetricsEngine:
    def __init__(self):
        self.rouge  = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smooth = SmoothingFunction().method1
        try:
            import nltk
            nltk.data.find("corpora/wordnet")
        except LookupError:
            import nltk
            nltk.download("wordnet",  quiet=True)
            nltk.download("omw-1.4", quiet=True)

    def compute_all(self, generated: str, reference: str) -> dict:
        gl, rl  = generated.lower(), reference.lower()
        gt_tok  = rl.split()
        gen_tok = gl.split()

        b1 = sentence_bleu([gt_tok], gen_tok, weights=(1,0,0,0),
                           smoothing_function=self.smooth)
        b2 = sentence_bleu([gt_tok], gen_tok, weights=(.5,.5,0,0),
                           smoothing_function=self.smooth)
        rs = self.rouge.score(rl, gl)
        mt = meteor_score([gt_tok], gen_tok)

        kh = sum(1 for kw in EVENT_KEYWORDS if kw in gl)
        kt = sum(1 for kw in EVENT_KEYWORDS if kw in rl)

        return {
            "BLEU-1":           round(b1, 4),
            "BLEU-2":           round(b2, 4),
            "ROUGE-1 F1":       round(rs["rouge1"].fmeasure, 4),
            "ROUGE-2 F1":       round(rs["rouge2"].fmeasure, 4),
            "ROUGE-L F1":       round(rs["rougeL"].fmeasure, 4),
            "METEOR":           round(mt, 4),
            "Keyword Hits":     kh,
            "Keyword Total GT": kt,
            "Keyword Hit Rate": round(kh / max(kt, 1), 4),
        }


# =============================================================================
# MODEL LOAD / UNLOAD
# =============================================================================

def load_model(model_id: str):
    print(f"\n{'='*60}")
    print(f"  Loading : {model_id}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map={"": device},          # matches reference exactly
        torch_dtype=torch.bfloat16,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    print(f"  device    : {device}")
    print(f"  dtype     : {next(model.parameters()).dtype}")
    print(f"  Processor : {type(processor).__name__}")
    return model, processor, device


def unload_model(model, processor):
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  ✓ GPU cache cleared")


# =============================================================================
# INFERENCE — native video_path mode
# =============================================================================

def run_inference(
    model,
    processor,
    device: str,
    video_path: str,
    start_time: float = None,
    end_time:   float = None,
    fps:        int   = DEFAULT_FPS,
    max_frames: int   = DEFAULT_MAX_FRAMES,
    prompt:     str   = PROMPT,
):
    """
    Matches the official VideoLLaMA3 LitServe reference exactly:
    - System role as a separate message (not add_system_prompt=True alone)
    - video_path nested dict → processor.load_video() via ffmpeg/decord
    - torch.inference_mode() (not no_grad)
    - pixel_values cast to bfloat16 after .to(device)
    Returns (response_text, proc_time_sec, gen_time_sec).
    """

    # Build video content dict — processor reads video_path key
    video_content = {
        "video_path": video_path,
        "fps":        fps,
        "max_frames": max_frames,
    }
    if start_time is not None:
        video_content["start_time"] = start_time
    if end_time is not None:
        video_content["end_time"]   = end_time

    # ── Conversation — reference pattern: system role + user role ────────
    conversation = [
        {
            "role":    "system",
            "content": "You are a helpful assistant.",   # reference uses this exactly
        },
        {
            "role": "user",
            "content": [
                {
                    "type":  "video",
                    "video": video_content,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]

    print(f"\n  Sampling frames [fps={fps}, max_frames={max_frames}]...")
    t_proc = time.time()

    # ── Process — reference uses inference_mode wrapping entire block ─────
    with torch.inference_mode():
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move to device first, then cast pixel_values — exact reference order
        inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        proc_time = time.time() - t_proc
        n_tokens  = inputs["input_ids"].shape[1]
        print(f"  ✓ Processed  {proc_time:.1f}s | Input tokens: {n_tokens}")

        print("  Generating...")
        t_gen = time.time()

        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,       # extended vs reference's 128 for fuller captions
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    gen_time = time.time() - t_gen
    n_new    = output_ids.shape[1] - n_tokens
    print(f"  ✓ Generated  {n_new} tokens in {gen_time:.1f}s")

    response = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0].strip()

    return response, proc_time, gen_time


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VideoLLaMA3 full-video inference + GT metrics"
    )
    parser.add_argument("--video",      default=VIDEO_INPUT_PATH,
                        help="Path to video file (.mp4)")
    parser.add_argument("--gt",         default=UCA_ANNOTATIONS_JSON,
                        help="Path to UCFCrime annotation JSON")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,
                        help="Directory to save results CSV")
    parser.add_argument("--start",      type=float, default=None,
                        help="Clip start time in seconds (optional)")
    parser.add_argument("--end",        type=float, default=None,
                        help="Clip end time in seconds (optional)")
    parser.add_argument("--fps",        type=int,   default=DEFAULT_FPS,
                        help=f"Frame sampling FPS (default: {DEFAULT_FPS})")
    parser.add_argument("--max_frames", type=int,   default=DEFAULT_MAX_FRAMES,
                        help=f"Max frames to sample (default: {DEFAULT_MAX_FRAMES})")
    parser.add_argument("--model",      default=MODEL_ID,
                        help=f"HuggingFace model ID (default: {MODEL_ID})")
    parser.add_argument("--prompt",     default=PROMPT,
                        help="Override the inference prompt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  VideoLLaMA3 — Full Video Path Test + Metrics")
    print("="*60)
    print(f"  Video      : {os.path.basename(args.video)}")
    print(f"  Model      : {args.model}")
    print(f"  FPS        : {args.fps}  |  Max frames : {args.max_frames}")
    if args.start is not None:
        print(f"  Window     : {args.start}s → {args.end}s")

    # ── Step 1: Ground truth ─────────────────────────────────────────────
    print("\n[1/4] Loading ground truth...")
    chunk_gt, full_gt = load_ground_truth(args.gt, args.video)
    has_gt = bool(full_gt)
    if has_gt:
        preview = full_gt[:120] + ("..." if len(full_gt) > 120 else "")
        print(f"  Full GT ({len(full_gt.split())} words): {preview}")
    else:
        print("  No GT found — metrics will be keyword-only")

    # ── Step 2: Load model ───────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    model, processor, device = load_model(args.model)

    # ── Step 3: Inference ────────────────────────────────────────────────
    print("\n[3/4] Running inference...")
    response, proc_time, gen_time = run_inference(
        model      = model,
        processor  = processor,
        device     = device,
        video_path = args.video,
        start_time = args.start,
        end_time   = args.end,
        fps        = args.fps,
        max_frames = args.max_frames,
        prompt     = args.prompt,
    )

    # ── Step 4: Metrics ──────────────────────────────────────────────────
    print("\n[4/4] Computing metrics...")
    me = MetricsEngine()

    if has_gt:
        metrics = me.compute_all(response, full_gt)
    else:
        kh      = sum(1 for kw in EVENT_KEYWORDS if kw in response.lower())
        metrics = {
            "Keyword Hits":     kh,
            "Keyword Hit Rate": "N/A (no GT)",
        }

    unload_model(model, processor)

    # ── Print results ─────────────────────────────────────────────────────
    W = 60
    print(f"\n{'='*W}")
    print("  GENERATED CAPTION")
    print(f"{'='*W}\n{response}\n")

    if has_gt:
        print(f"{'─'*W}")
        print("  GROUND TRUTH")
        print(f"{'─'*W}\n{full_gt}\n")

    print(f"{'─'*W}")
    print("  METRICS")
    print(f"{'─'*W}")
    col = 22
    for k, v in metrics.items():
        if isinstance(v, float) and 0.0 <= v <= 1.0:
            filled = int(v * 20)
            bar    = f"[{'█' * filled}{'░' * (20 - filled)}] {v:.4f}"
            print(f"  {k:<{col}} {bar}")
        else:
            print(f"  {k:<{col}} {v}")

    print(f"\n{'─'*W}")
    print("  TIMING")
    print(f"{'─'*W}")
    print(f"  Frame sampling  : {proc_time:.2f}s")
    print(f"  Generation      : {gen_time:.2f}s")
    print(f"  Total           : {proc_time + gen_time:.2f}s")

    # ── Save CSV ─────────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "Model":            args.model,
        "Input Mode":       "VIDEO_FULL_PATH",
        "Video":            os.path.basename(args.video),
        "Start (s)":        args.start,
        "End (s)":          args.end,
        "FPS":              args.fps,
        "Max Frames":       args.max_frames,
        "Caption":          response,
        "Ground Truth":     full_gt if has_gt else "N/A",
        "Has GT":           has_gt,
        "Proc Time (s)":    round(proc_time, 3),
        "Gen Time (s)":     round(gen_time,  3),
        "Total Time (s)":   round(proc_time + gen_time, 3),
        **metrics,
    }
    out_path = os.path.join(
        args.output_dir, f"videollama3_fullvideo_{ts}.csv"
    )
    pd.DataFrame([row]).to_csv(out_path, index=False)

    print(f"\n  ✓ Saved → {out_path}")
    print("="*W + "\n")


if __name__ == "__main__":
    main()
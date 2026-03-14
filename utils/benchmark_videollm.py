"""
AgentVigil — VLM + VideoLLM Benchmarking Pipeline
===================================================

Extract raw video clip → pass as video tensor

This directly answers: "Does native video input beat frame extraction?"

Usage:
    python benchmark_vlm.py

Requirements:
    pip install torch transformers==4.45.0 accelerate==0.34.0 bitsandbytes==0.43.0
    pip install pillow opencv-python pandas nltk rouge-score
    pip install qwen-vl-utils einops timm av
"""

import os
import re
import cv2
import json
import time
import gc
import warnings
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

VIDEO_INPUT_PATH = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCF_Crimes\UCF_Crimes\Videos\Vandalism\Vandalism029_x264.mp4"
UCA_ANNOTATIONS_JSON = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Train.json"
OUTPUT_DIR = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\benchmark_results"
DEBUG_FRAMES_DIR = os.path.join(OUTPUT_DIR, "debug_frames")
DEBUG_FRAMES_DIR = os.path.join(OUTPUT_DIR, "debug_video")

MAX_CHUNKS_FOR_BENCHMARK = 50
FRAMES_PER_SECOND = 1
VIDEO_MAX_FRAMES_PER_CLIP = 16

CHUNK_DURATION_TIERS = [
    (30, 2), (120, 3), (300, 5), (1800, 10), (float('inf'), 15),
]

# =============================================================================
# MODEL LINEUP — Image Mode + Video Mode
# =============================================================================

MODELS_TO_BENCHMARK = {
    "Qwen2-VL-7B [VID]":           ("Qwen2VLVideoAdapter",       "Qwen/Qwen2-VL-7B-Instruct"),
    "Qwen2.5-VL-7B [VID]":         ("Qwen25VLVideoAdapter",      "Qwen/Qwen2.5-VL-7B-Instruct"),
    "Qwen2.5-VL-3B [VID]":         ("Qwen25VLVideoAdapter",      "Qwen/Qwen2.5-VL-3B-Instruct"),
    "LLaVA-OneVision-7B [VID]":    ("LLaVAOneVisionVidAdapter",  "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "VideoLLaMA3-2B [VID]":        ("VideoLLaMA3VideoAdapter",   "DAMO-NLP-SG/VideoLLaMA3-2B")
}

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
# SURVEILLANCE PROMPT
# =============================================================================

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

Example 3 — Normal Activity:
"Pedestrians walk along a sidewalk at a steady pace. Vehicles move through the intersection following traffic signals. No unusual or anomalous behavior is detected. EVENT: Normal Activity."

Now analyze the following surveillance frames:"""


def build_user_prompt(num_frames: int, is_video: bool = False) -> str:
    media = "video clip" if is_video else "sequential frames"
    return (
        f"These are {num_frames} frames from surveillance/CCTV footage "
        f"({'passed as a video clip' if is_video else 'extracted as images'}). "
        f"They represent a short time window of activity.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Analyze the temporal progression across all {num_frames} frames. "
        f"Describe the specific event occurring, how it develops from frame to frame, "
        f"and classify it. Be precise and action-oriented."
    )


# =============================================================================
# VIDEO CLIP EXTRACTOR
# =============================================================================

def extract_video_clip(video_path, start_sec, end_sec, max_frames=VIDEO_MAX_FRAMES_PER_CLIP):
    """Extract raw RGB frames as np.ndarray (T, H, W, 3) for VideoLLMs."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)
    total_clip = end_frame - start_frame

    if total_clip <= max_frames:
        indices = list(range(start_frame, end_frame))
    else:
        step = total_clip / max_frames
        indices = [int(start_frame + i * step) for i in range(max_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else np.array([])


# =============================================================================
# CHUNKING & FRAME EXTRACTION
# =============================================================================

def determine_chunk_duration(total_duration_sec):
    for threshold, duration in CHUNK_DURATION_TIERS:
        if total_duration_sec <= threshold:
            return duration
    return CHUNK_DURATION_TIERS[-1][1]

def select_chunk_indices(total_chunks, max_chunks, strategy="uniform"):
    if total_chunks <= max_chunks:
        return list(range(total_chunks))
    step = total_chunks / max_chunks
    return [int(i * step) for i in range(max_chunks)]

def extract_chunks_and_frames(video_path, output_dir, chunk_duration_sec=None,
                               max_chunks=MAX_CHUNKS_FOR_BENCHMARK, sampling_strategy="uniform"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS")
    total_duration = total_frames / fps

    if chunk_duration_sec is None:
        chunk_duration_sec = determine_chunk_duration(total_duration)

    total_possible = max(1, int(total_duration / chunk_duration_sec))
    if total_duration % chunk_duration_sec >= 0.5:
        total_possible += 1

    is_sampled = total_possible > max_chunks
    selected = set(select_chunk_indices(total_possible, max_chunks, sampling_strategy))

    print(f"[VIDEO] {video_name} | {fps:.0f}fps | {total_duration:.1f}s | "
          f"Chunk: {chunk_duration_sec}s | Selected: {len(selected)}/{total_possible}")

    chunks = {}
    chunk_index = 0
    current_start = 0.0
    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration_sec, total_duration)
        if chunk_end - current_start < 0.5:
            break
        if chunk_index not in selected:
            chunk_index += 1
            current_start += chunk_duration_sec
            continue

        num_frames = max(1, int(chunk_end - current_start))
        saved_paths = []
        for i in range(num_frames):
            target_idx = min(int((current_start + i) * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            success, frame = cap.read()
            if success:
                fname = f"{video_name}_c{chunk_index:04d}_f{i}.jpg"
                fpath = os.path.join(frames_dir, fname)
                cv2.imwrite(fpath, frame)
                saved_paths.append(fpath)

        if saved_paths:
            chunks[chunk_index] = {
                "frame_paths": saved_paths,
                "start_sec": round(current_start, 2),
                "end_sec": round(chunk_end, 2),
                "num_frames": len(saved_paths),
                "chunk_duration_sec": chunk_duration_sec,
                "is_sampled": is_sampled,
            }
        chunk_index += 1
        current_start += chunk_duration_sec

    cap.release()
    print(f"  Extracted: {len(chunks)} chunks | "
          f"{sum(c['num_frames'] for c in chunks.values())} frames")
    return chunks


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_ground_truth(json_path, video_path, chunk_duration_sec):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    short_name = "_".join(video_name.split("_")[:-1])
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    info = None
    for key in [video_name, short_name]:
        if key in annotations:
            info = annotations[key]
            break
    if not info:
        return {}

    timestamps, sentences = info.get("timestamps", []), info.get("sentences", [])
    chunk_gt, chunk_idx, current_start = {}, 0, 0.0
    while current_start < 3600:
        chunk_end = current_start + chunk_duration_sec
        matched = [s for (ts, te), s in zip(timestamps, sentences)
                   if current_start < te and ts < chunk_end]
        if matched:
            chunk_gt[chunk_idx] = " ".join(matched)
        chunk_idx += 1
        current_start += chunk_duration_sec
    print(f"[GT] {len(chunk_gt)} chunks with annotations")
    return chunk_gt


# =============================================================================
# METRICS
# =============================================================================

class MetricsEngine:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        try:
            import nltk; nltk.data.find('corpora/wordnet')
        except LookupError:
            import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)

    def compute_all(self, gen, ref):
        gl, rl = gen.lower(), ref.lower()
        rt, gt_ = rl.split(), gl.split()
        b1 = sentence_bleu([rt], gt_, weights=(1,0,0,0), smoothing_function=self.smoothing)
        b2 = sentence_bleu([rt], gt_, weights=(.5,.5,0,0), smoothing_function=self.smoothing)
        rs = self.rouge.score(rl, gl)
        mt = meteor_score([rt], gt_)
        kh = sum(1 for kw in EVENT_KEYWORDS if kw in gl)
        kt = sum(1 for kw in EVENT_KEYWORDS if kw in rl)
        return {
            "BLEU-1": round(b1,4), "BLEU-2": round(b2,4),
            "ROUGE-1 F1": round(rs['rouge1'].fmeasure,4),
            "ROUGE-2 F1": round(rs['rouge2'].fmeasure,4),
            "ROUGE-L F1": round(rs['rougeL'].fmeasure,4),
            "METEOR": round(mt,4),
            "Keyword Hits": kh, "Keyword Total (GT)": kt,
            "Keyword Hit Rate": round(kh/max(kt,1), 4),
        }


# =============================================================================
# BASE ADAPTER
# =============================================================================

class BaseAdapter(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_mode = "image"

    @abstractmethod
    def load(self): pass
    @abstractmethod
    def generate_caption(self, **kwargs) -> Tuple[str, float]: pass
    @abstractmethod
    def unload(self): pass

    def _cleanup_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# =============================================================================
# GENERIC VIDEO LLM ADAPTER (InternVL / VideoLLaMA / MiniCPM / InternVideo)
# =============================================================================

class GenericHFVideoAdapter(BaseAdapter):
    """
    Universal HuggingFace VideoLLM adapter.

    Works for:
        - VideoLLaMA3
        - InternVL
        - InternVideo
        - MiniCPM-V

    Uses AutoModelForCausalLM + AutoProcessor
    """

    def __init__(self, model_id):
        super().__init__(model_id)
        self.input_mode = "video"

    def load(self):
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"  [LOAD] {self.model_id} 🎬 VIDEO")

        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=qconfig,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

    def generate_caption(self, video_clip=None, num_frames=None, **kw):

        n = num_frames or len(video_clip)
        frames = [Image.fromarray(f) for f in video_clip]

        prompt = build_user_prompt(n, is_video=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        start = time.time()

        inputs = self.processor(
            text=[text],
            videos=[frames],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        latency = time.time() - start

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, outputs)
        ]

        caption = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True
        )[0].strip()

        return caption, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()

# =============================================================================
# MODE B: VIDEO ADAPTERS
# =============================================================================

class Qwen2VLVideoAdapter(BaseAdapter):
    """Qwen2-VL-7B in VIDEO mode — passes frames as video sequence."""
    def __init__(self, mid):
        super().__init__(mid); self.input_mode = "video"
    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id} 🎬 VIDEO")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=q, device_map="auto", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    def generate_caption(self, video_clip=None, num_frames=None, **kw):
        n = num_frames or len(video_clip)
        pil_frames = [Image.fromarray(f) for f in video_clip]
        # Pass as a video sequence — all frames under one "video" context
        video_content = [{"type": "video", "video": pil_frames}]
        uc = video_content + [{"type": "text", "text": build_user_prompt(n, is_video=True)}]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": uc}]
        ti = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = time.time()
        inp = self.processor(text=[ti], images=None, videos=[pil_frames], padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=150, do_sample=False)
        lat = time.time() - t
        tr = [o[len(i):] for i, o in zip(inp.input_ids, g)]
        cap = self.processor.batch_decode(tr, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return cap, lat
    def unload(self):
        del self.model, self.processor; self._cleanup_gpu()


class Qwen25VLVideoAdapter(BaseAdapter):
    """Qwen2.5-VL (3B or 7B) in VIDEO mode."""
    def __init__(self, mid):
        super().__init__(mid); self.input_mode = "video"
    def load(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id} 🎬 VIDEO")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=q, device_map="auto", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    def generate_caption(self, video_clip=None, num_frames=None, **kw):
        n = num_frames or len(video_clip)
        pil_frames = [Image.fromarray(f) for f in video_clip]
        video_content = [{"type": "video", "video": pil_frames}]
        uc = video_content + [{"type": "text", "text": build_user_prompt(n, is_video=True)}]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": uc}]
        ti = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = time.time()
        inp = self.processor(text=[ti], images=None, videos=[pil_frames], padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=150, do_sample=False)
        lat = time.time() - t
        tr = [o[len(i):] for i, o in zip(inp.input_ids, g)]
        cap = self.processor.batch_decode(tr, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return cap, lat
    def unload(self):
        del self.model, self.processor; self._cleanup_gpu()

class VideoLLaMA3VideoAdapter(BaseAdapter):
    """
    VideoLLaMA3 (2B/7B) — matches official inference example exactly.
    Source: github.com/DAMO-NLP-SG/VideoLLaMA3/inference/example_videollama3.py
    """

    def __init__(self, mid):
        super().__init__(mid)
        self.input_mode = "video"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        print(f"  [LOAD] {self.model_id} 🎬 VIDEO")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        print("  Processor:", type(self.processor))

    def generate_caption(self, video_clip=None, num_frames=None, **kw):

        # ── 1. Cap at 16 frames ─────────────────────────────────────────
        if len(video_clip) > 16:
            idx = np.linspace(0, len(video_clip) - 1, 16).astype(int)
            video_clip = video_clip[idx]

        # ── 2. Channels-first (C,H,W) uint8 — what the image processor needs
        frames_chw = []
        for f in video_clip:
            arr = f if f.dtype == np.uint8 else (f * 255).astype(np.uint8)
            frames_chw.append(arr.transpose(2, 0, 1))  # (H,W,3) → (3,H,W)

        n = len(frames_chw)

        # Neutral prompt that doesn't trigger safety filters
        VIDEOLLAMA3_SYSTEM_PROMPT = """You are a video analysis assistant. 
        Describe what is happening in the video frames in detail.
        Focus on: the people present, their actions and movements, 
        any objects involved, and how the scene changes over time.
        Be specific and factual."""

        VIDEOLLAMA3_USER_PROMPT = """Analyze these surveillance video frames.
        Describe the activity shown: who is present, what actions are occurring, 
        and how the scene progresses. If any unusual or notable activity is visible, 
        describe it precisely."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames_chw,   # List of (C,H,W) np.ndarray
                        "num_frames": n,
                    },
                    {
                        "type": "text",
                        "text": f"{VIDEOLLAMA3_SYSTEM_PROMPT}\n\n{VIDEOLLAMA3_USER_PROMPT}"
                    }
                ]
            }
        ]

        # ── 4. Exact call signature from official example ────────────────
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,        # ← was missing / wrong before
            add_generation_prompt=True,
            return_tensors="pt"
        )

        # ── 5. Move to device — ONLY cast pixel_values to bfloat16 ───────
        #       Official example does exactly this: move all tensors to cuda,
        #       then specifically cast pixel_values dtype separately.
        inputs = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # ── 6. Generate ──────────────────────────────────────────────────
        t = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        latency = time.time() - t

        # ── 7. Decode full output — official example does NOT trim input prefix
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        return response, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()

class LLaVAOneVisionVidAdapter(BaseAdapter):
    """LLaVA-OneVision in VIDEO mode — uses 'video' type in messages."""
    def __init__(self, mid):
        super().__init__(mid); self.input_mode = "video"
    def load(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id} 🎬 VIDEO")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=q, device_map="auto", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    def generate_caption(self, video_clip=None, num_frames=None, **kw):
        n = num_frames or len(video_clip)
        pil_frames = [Image.fromarray(f) for f in video_clip]
        uc = [{"type": "video"}, {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{build_user_prompt(n, is_video=True)}"}]
        msgs = [{"role": "user", "content": uc}]
        ti = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        t = time.time()
        inp = self.processor(text=ti, videos=[pil_frames], return_tensors="pt").to(self.model.device, dtype=torch.float16)
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=150, do_sample=False)
        lat = time.time() - t
        tr = [o[len(i):] for i, o in zip(inp.input_ids, g)]
        cap = self.processor.batch_decode(tr, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return cap, lat
    def unload(self):
        del self.model, self.processor; self._cleanup_gpu()


# =============================================================================
# ADAPTER REGISTRY
# =============================================================================

ADAPTER_CLASSES = {

    # Qwen
    "Qwen2VLVideoAdapter": Qwen2VLVideoAdapter,
    "Qwen25VLVideoAdapter": Qwen25VLVideoAdapter,
    "Qwen25VLVideoAdapter": Qwen25VLVideoAdapter,

    # LLaVA
    "LLaVAOneVisionVidAdapter": LLaVAOneVisionVidAdapter,

    # Generic VideoLLM
    "VideoLLaMA3VideoAdapter": VideoLLaMA3VideoAdapter,
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)

    print("=" * 80)
    print("  AgentVigil — VLM + VideoLLM Benchmark")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Video: {os.path.basename(VIDEO_INPUT_PATH)}")
    print("=" * 80)

    # Step 1: Extract chunks
    print("\n[STEP 1/5] Extracting chunks...")
    chunks = extract_chunks_and_frames(VIDEO_INPUT_PATH, DEBUG_FRAMES_DIR)
    if not chunks:
        print("[FATAL] No chunks."); return

    eff_dur = next(iter(chunks.values()))["chunk_duration_sec"]

    # Step 2: Ground truth
    print("\n[STEP 2/5] Loading ground truth...")
    gt_map = load_ground_truth(UCA_ANNOTATIONS_JSON, VIDEO_INPUT_PATH, eff_dur)

    # Step 3: Metrics
    print("\n[STEP 3/5] Initializing metrics...")
    me = MetricsEngine()

    # Step 4: Benchmark
    print("\n[STEP 4/5] Running benchmarks...\n")
    detailed, summaries = [], []

    for model_name, (adapter_cls_name, model_id) in MODELS_TO_BENCHMARK.items():
        print(f"\n{'─' * 70}")
        print(f"  {model_name} ({model_id})")
        print(f"{'─' * 70}")

        AdCls = ADAPTER_CLASSES.get(adapter_cls_name)
        if not AdCls:
            print(f"  [ERROR] Unknown adapter: {adapter_cls_name}"); continue

        adapter = AdCls(model_id)
        try:
            adapter.load()
        except Exception as e:
            print(f"  [ERROR] Load failed: {e}"); adapter._cleanup_gpu(); continue

        is_vid = adapter.input_mode == "video"
        lats, mets = [], {k: [] for k in ["BLEU-1","BLEU-2","ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1","METEOR","Keyword Hit Rate"]}
        gt_count, proc_count = 0, 0
        t0 = time.time()

        for cidx, cinfo in sorted(chunks.items()):
            try:
                if is_vid:
                    clip = extract_video_clip(VIDEO_INPUT_PATH, cinfo["start_sec"], cinfo["end_sec"])
                    if len(clip) == 0: raise RuntimeError("Empty clip")
                    cap, lat = adapter.generate_caption(video_clip=clip, num_frames=len(clip))
                else:
                    cap, lat = adapter.generate_caption(frame_paths=cinfo["frame_paths"])
            except Exception as e:
                print(f"  [ERROR] Chunk {cidx}: {e}"); cap, lat = f"[ERROR] {str(e)[:80]}", 0.0

            lats.append(lat); proc_count += 1
            gt = gt_map.get(cidx, ""); has_gt = bool(gt)
            md = {}
            if has_gt:
                md = me.compute_all(cap, gt); gt_count += 1
                for k in mets:
                    if k in md: mets[k].append(md[k])

            detailed.append({
                "Model": model_name, "Model ID": model_id,
                "Input Mode": "VIDEO" if is_vid else "IMAGE",
                "Chunk": cidx, "Start (s)": cinfo["start_sec"], "End (s)": cinfo["end_sec"],
                "Duration (s)": cinfo["chunk_duration_sec"], "Frames": cinfo["num_frames"],
                "Caption": cap, "Ground Truth": gt if has_gt else "N/A",
                "Has GT": has_gt, "Latency (s)": round(lat, 3),
                **{k: md.get(k, "") for k in ["BLEU-1","BLEU-2","ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1","METEOR","Keyword Hits","Keyword Total (GT)","Keyword Hit Rate"]},
            })

            tag = "🎬" if is_vid else "🖼️"
            st = f"KHR={md.get('Keyword Hit Rate','N/A')}" if has_gt else "no GT"
            print(f"  {tag} Chunk {cidx:3d} | {cinfo['start_sec']:5.1f}-{cinfo['end_sec']:5.1f}s | "
                  f"Lat: {lat:.1f}s | {st}")
            print(f"     {cap[:100]}...")

        tt = time.time() - t0
        al = sum(lats)/len(lats) if lats else 0
        s = {
            "Model": model_name, "Model ID": model_id,
            "Input Mode": "VIDEO" if is_vid else "IMAGE",
            "Chunks": proc_count, "GT Chunks": gt_count,
            "Duration (s)": eff_dur, "Total Time (s)": round(tt,2),
            "Avg Latency (s)": round(al,3),
            "Min Latency (s)": round(min(lats),3) if lats else 0,
            "Max Latency (s)": round(max(lats),3) if lats else 0,
            "Throughput (c/min)": round(proc_count/tt*60,1) if tt>0 else 0,
        }
        for mn, vals in mets.items():
            s[f"Avg {mn}"] = round(sum(vals)/len(vals),4) if vals else "N/A"
        summaries.append(s)
        print(f"\n  ✓ {model_name} done: {proc_count} chunks in {tt:.1f}s ({al:.1f}s avg)")
        adapter.unload()

    # Step 5: Export
    print(f"\n{'=' * 80}\n[STEP 5/5] Exporting...")
    dp = os.path.join(OUTPUT_DIR, f"benchmark_detailed_{timestamp}.csv")
    pd.DataFrame(detailed).to_csv(dp, index=False)
    sp = os.path.join(OUTPUT_DIR, f"benchmark_summary_{timestamp}.csv")
    df_s = pd.DataFrame(summaries)
    df_s.to_csv(sp, index=False)

    print(f"\n{'=' * 80}\n  BENCHMARK SUMMARY\n{'=' * 80}\n")
    print(df_s.to_string(index=False))

    # IMAGE vs VIDEO comparison
    print(f"\n{'─' * 70}\n  🖼️ IMAGE vs 🎬 VIDEO — Head-to-Head\n{'─' * 70}")
    cols = ["Model", "Input Mode", "Avg Latency (s)", "Throughput (c/min)",
            "Avg BLEU-2", "Avg ROUGE-L F1", "Avg METEOR", "Avg Keyword Hit Rate"]
    avail = [c for c in cols if c in df_s.columns]
    print(df_s[avail].to_string(index=False))

    # Mode averages
    print(f"\n{'─' * 70}\n  MODE AVERAGES\n{'─' * 70}")
    for mode in ["IMAGE", "VIDEO"]:
        mr = df_s[df_s["Input Mode"] == mode]
        if len(mr) > 0:
            al = mr["Avg Latency (s)"].mean()
            kv = mr["Avg Keyword Hit Rate"].apply(lambda x: float(x) if x != "N/A" else 0)
            ak = kv.mean()
            print(f"  {mode:6s} | Avg Latency: {al:.2f}s | Avg KHR: {ak:.4f}")

    print(f"\n{'=' * 80}")
    print(f"  Complete! Detailed: {dp}")
    print(f"  Summary: {sp}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
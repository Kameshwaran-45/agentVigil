"""
AgentVigil — VLM Benchmarking Pipeline (Adaptive)
===================================================
Benchmarks multiple Vision-Language Models on surveillance video chunks.
Adapts chunk size and sampling strategy based on video length to keep
benchmark time manageable for any video duration.

Produces:
  1. benchmark_detailed_<timestamp>.csv  — Per (model × chunk) results
  2. benchmark_summary_<timestamp>.csv   — Per model averaged metrics

Usage:
    python benchmark_vlm.py

Requirements:
    pip install torch transformers accelerate bitsandbytes
    pip install sentence-transformers pillow opencv-python pandas
    pip install nltk rouge-score qwen-vl-utils einops timm
"""

import os
import re
import cv2
import json
import time
import gc
import warnings
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

# --- Paths ---
VIDEO_INPUT_PATH = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCF_Crimes\UCF_Crimes\Videos\Shoplifting\Shoplifting033_x264.mp4"
UCA_ANNOTATIONS_JSON = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\UCFCrime_Train.json"
OUTPUT_DIR = r"C:\Users\Kameshwaran\Downloads\pes\Capstone\dataset\benchmark_results"
DEBUG_FRAMES_DIR = os.path.join(OUTPUT_DIR, "debug_frames")

# --- Adaptive Chunking Configuration ---
MAX_CHUNKS_FOR_BENCHMARK = 50       # Cap total chunks to keep demo feasible
FRAMES_PER_SECOND = 1               # Extraction density: 1 frame per second

# Adaptive chunk duration tiers (video_length_threshold_sec, chunk_duration_sec)
CHUNK_DURATION_TIERS = [
    (30,           3),     # Videos <= 30s   -> 2s chunks (dense)
    (120,          4),     # Videos <= 2min  -> 3s chunks (default)
    (300,          6),     # Videos <= 5min  -> 5s chunks
    (1800,         10),    # Videos <= 30min -> 10s chunks
    (float('inf'), 15),   # Videos > 30min  -> 15s chunks
]

# --- Models to Benchmark ---
MODELS_TO_BENCHMARK = {
    "LLaVA-NeXT-7B":        ("LLaVANextAdapter",     "llava-hf/llava-v1.6-mistral-7b-hf"),
    "Qwen2-VL-7B-Instruct": ("Qwen2VLAdapter",       "Qwen/Qwen2-VL-7B-Instruct"),
    # "InternVL2-8B":         ("InternVL2Adapter",      "OpenGVLab/InternVL2-8B"),
    "Phi-3.5-Vision":       ("Phi35VisionAdapter",    "microsoft/Phi-3.5-vision-instruct"),
    "MiniCPM-V-4.5":        ("MiniCPMV45Adapter",     "openbmb/MiniCPM-V-4_5"),
    "LLaVA-OneVision-7B":   ("LLaVAOneVisionAdapter", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "Qwen2.5-VL-7B-Instruct":         ("Qwen25VLImageAdapter",    "Qwen/Qwen2.5-VL-7B-Instruct"),
    "Qwen2.5-VL-3B-Instruct":         ("Qwen25VLImageAdapter",    "Qwen/Qwen2.5-VL-3B-Instruct"),

}

# --- Surveillance Event Keywords (for keyword-hit metric) ---
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
# SURVEILLANCE PROMPT (shared across all models)
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

Example 3 — Fighting:
"Two men face each other aggressively in a parking lot. In the initial frames, one pushes the other. The confrontation escalates as both begin throwing punches. Bystanders start moving away. EVENT: Fighting / Assault."

Example 4 — Shoplifting:
"A person in a store aisle looks around, then quickly places merchandise into their jacket pocket. They then walk briskly toward the exit without approaching the checkout counter. EVENT: Shoplifting / Stealing."

Example 5 — Vandalism:
"An individual approaches a parked car and uses a blunt object to smash the side window. Glass fragments scatter. The person then moves to the next vehicle. EVENT: Vandalism / Property Damage."

Example 6 — Normal Activity:
"Pedestrians walk along a sidewalk at a steady pace. Vehicles move through the intersection following traffic signals. No unusual or anomalous behavior is detected. EVENT: Normal Activity."

Now analyze the following surveillance frames:"""


def build_user_prompt(num_frames: int) -> str:
    """Builds the user-facing analysis prompt with few-shot examples."""
    return (
        f"These are {num_frames} sequential frames extracted from surveillance/CCTV footage. "
        f"They represent a short time window of activity.\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Analyze the temporal progression across all {num_frames} frames. "
        f"Describe the specific event occurring, how it develops from frame to frame, "
        f"and classify it. Be precise and action-oriented."
    )


# =============================================================================
# ADAPTIVE VIDEO CHUNKING & FRAME EXTRACTION
# =============================================================================

def determine_chunk_duration(total_duration_sec: float) -> int:
    """
    Selects the optimal chunk duration based on video length.
    Longer videos get wider chunks to keep total workload manageable.
    """
    for threshold, duration in CHUNK_DURATION_TIERS:
        if total_duration_sec <= threshold:
            return duration
    return CHUNK_DURATION_TIERS[-1][1]


def select_chunk_indices(
    total_chunks: int,
    max_chunks: int,
    strategy: str = "uniform"
) -> List[int]:
    """
    When total chunks exceed the budget, selects a representative subset.

    Strategies:
        - "uniform":  Evenly spaced across the video (default, best for general use)
        - "edges":    Prioritize start + end (events often begin/conclude at edges)
        - "all":      No sampling, return everything (ignore max_chunks)
    """
    if total_chunks <= max_chunks or strategy == "all":
        return list(range(total_chunks))

    if strategy == "uniform":
        step = total_chunks / max_chunks
        return [int(i * step) for i in range(max_chunks)]

    elif strategy == "edges":
        edge_count = max_chunks // 3
        middle_count = max_chunks - 2 * edge_count
        start_indices = list(range(edge_count))
        end_indices = list(range(total_chunks - edge_count, total_chunks))
        middle_start = edge_count
        middle_end = total_chunks - edge_count
        if middle_count > 0 and middle_end > middle_start:
            middle_step = (middle_end - middle_start) / middle_count
            middle_indices = [int(middle_start + i * middle_step) for i in range(middle_count)]
        else:
            middle_indices = []
        return sorted(set(start_indices + middle_indices + end_indices))

    return list(range(min(total_chunks, max_chunks)))


def extract_chunks_and_frames(
    video_path: str,
    output_dir: str,
    chunk_duration_sec: int = None,
    max_chunks: int = MAX_CHUNKS_FOR_BENCHMARK,
    sampling_strategy: str = "uniform",
) -> Dict[int, Dict[str, Any]]:
    """
    Adaptively splits video into chunks and extracts 1 frame/sec from each.

    Behavior adapts to video length:
        - Short videos:  Dense small chunks (miss nothing)
        - Long videos:   Wider chunks + sampling (stay within time budget)

    Args:
        video_path:          Path to input video.
        output_dir:          Where to save extracted frame JPEGs.
        chunk_duration_sec:  Override chunk size. None = auto-detect from duration.
        max_chunks:          Maximum chunks to process. Excess are sampled.
        sampling_strategy:   "uniform", "edges", or "all".

    Returns:
        {chunk_index: {
            "frame_paths": List[str],
            "start_sec": float,
            "end_sec": float,
            "num_frames": int,
            "chunk_duration_sec": int,
            "is_sampled": bool
        }}
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    # --- Open video and read properties ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    total_duration = total_frames / fps

    # --- ADAPTIVE: Determine chunk duration ---
    if chunk_duration_sec is None:
        chunk_duration_sec = determine_chunk_duration(total_duration)

    # --- Calculate total possible chunks ---
    total_possible_chunks = max(1, int(total_duration / chunk_duration_sec))
    if total_duration % chunk_duration_sec >= 0.5:
        total_possible_chunks += 1

    # --- ADAPTIVE: Sample chunks if too many ---
    is_sampled = total_possible_chunks > max_chunks
    selected_indices = select_chunk_indices(
        total_possible_chunks, max_chunks, sampling_strategy
    )
    selected_set = set(selected_indices)

    print(f"[VIDEO] {video_name}")
    print(f"  FPS: {fps:.1f} | Duration: {total_duration:.1f}s | Total Frames: {total_frames}")
    print(f"  Adaptive Chunk Duration: {chunk_duration_sec}s "
          f"(auto-selected for {total_duration:.0f}s video)")
    print(f"  Total Possible Chunks: {total_possible_chunks}")
    print(f"  Chunks Selected: {len(selected_indices)}"
          f"{' (SAMPLED)' if is_sampled else ' (all)'}"
          f" | Strategy: {sampling_strategy}")

    # --- Extract frames chunk by chunk ---
    chunks = {}
    chunk_index = 0
    current_start = 0.0

    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration_sec, total_duration)
        actual_duration = chunk_end - current_start

        if actual_duration < 0.5:
            break

        # Skip chunks not in our selected sample
        if chunk_index not in selected_set:
            chunk_index += 1
            current_start += chunk_duration_sec
            continue

        # --- Extract 1 frame per second within this chunk ---
        num_frames = max(1, int(actual_duration))
        saved_paths = []

        for i in range(num_frames):
            target_time = current_start + i
            target_frame_idx = int(target_time * fps)
            target_frame_idx = min(target_frame_idx, total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            success, frame = cap.read()
            if not success:
                continue

            fname = f"{video_name}_chunk{chunk_index:04d}_frame{i}.jpg"
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

    total_extracted_frames = sum(c["num_frames"] for c in chunks.values())
    print(f"  Result: {len(chunks)} chunks | {total_extracted_frames} total frames "
          f"| Saved to: {frames_dir}")

    return chunks


# =============================================================================
# GROUND TRUTH LOADER
# =============================================================================

def load_ground_truth(
    json_path: str,
    video_path: str,
    chunk_duration_sec: int
) -> Dict[int, str]:
    """
    Loads annotation JSON and maps each chunk index to its ground truth
    sentence if the chunk overlaps with an annotated event timestamp.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    short_name = "_".join(video_name.split("_")[:-1])

    if not os.path.exists(json_path):
        print(f"[WARN] Annotation file not found: {json_path}")
        return {}

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    # Find the matching annotation key
    info = None
    for key_candidate in [video_name, short_name]:
        if key_candidate in annotations:
            info = annotations[key_candidate]
            break

    if info is None:
        print(f"[WARN] No annotation found for '{video_name}' in JSON.")
        return {}

    timestamps = info.get("timestamps", [])
    sentences = info.get("sentences", [])

    chunk_gt = {}
    chunk_idx = 0
    current_start = 0.0

    while current_start < 3600:  # max 1 hour
        chunk_end = current_start + chunk_duration_sec
        matched_sentences = []

        for (t_start, t_end), sentence in zip(timestamps, sentences):
            if current_start < t_end and t_start < chunk_end:
                matched_sentences.append(sentence)

        if matched_sentences:
            chunk_gt[chunk_idx] = " ".join(matched_sentences)

        chunk_idx += 1
        current_start += chunk_duration_sec

    print(f"[ANNOTATIONS] Found ground truth for {len(chunk_gt)} chunks.")
    return chunk_gt


# =============================================================================
# METRICS ENGINE
# =============================================================================

class MetricsEngine:
    """Computes NLP metrics between generated caption and ground truth."""

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.smoothing = SmoothingFunction().method1
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download required NLTK data once."""
        try:
            import nltk
            nltk.data.find('corpora/wordnet')
        except LookupError:
            import nltk
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    def compute_all(self, generated: str, reference: str) -> Dict[str, float]:
        """Returns a dict of all metric scores."""
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        ref_tokens = ref_lower.split()
        gen_tokens = gen_lower.split()

        # --- BLEU ---
        bleu_1 = sentence_bleu(
            [ref_tokens], gen_tokens,
            weights=(1.0, 0, 0, 0),
            smoothing_function=self.smoothing
        )
        bleu_2 = sentence_bleu(
            [ref_tokens], gen_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=self.smoothing
        )

        # --- ROUGE ---
        rouge_scores = self.rouge.score(ref_lower, gen_lower)

        # --- METEOR ---
        meteor = meteor_score([ref_tokens], gen_tokens)

        # --- Keyword Hit Rate ---
        keyword_hits = sum(1 for kw in EVENT_KEYWORDS if kw in gen_lower)
        keyword_total = sum(1 for kw in EVENT_KEYWORDS if kw in ref_lower)
        keyword_hit_rate = keyword_hits / max(keyword_total, 1)

        return {
            "BLEU-1": round(bleu_1, 4),
            "BLEU-2": round(bleu_2, 4),
            "ROUGE-1 F1": round(rouge_scores['rouge1'].fmeasure, 4),
            "ROUGE-2 F1": round(rouge_scores['rouge2'].fmeasure, 4),
            "ROUGE-L F1": round(rouge_scores['rougeL'].fmeasure, 4),
            "METEOR": round(meteor, 4),
            "Keyword Hits": keyword_hits,
            "Keyword Total (GT)": keyword_total,
            "Keyword Hit Rate": round(keyword_hit_rate, 4),
        }


# =============================================================================
# MODEL ADAPTERS (Unified Interface)
# =============================================================================

class BaseVLMAdapter(ABC):
    """Abstract base class for all VLM model adapters."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load(self):
        """Load model and processor into memory."""
        pass

    @abstractmethod
    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        """
        Generate a surveillance caption from frame image paths.
        Returns: (caption_string, latency_seconds)
        """
        pass

    @abstractmethod
    def unload(self):
        """Free GPU memory."""
        pass

    def _cleanup_gpu(self):
        """Force garbage collection and clear CUDA cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- Adapter 1: LLaVA-NeXT (v1.6) Mistral-7B ---

class LLaVANextAdapter(BaseVLMAdapter):

    def load(self):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id}")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=qconfig,
            device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = build_user_prompt(num)

        # LLaVA-NeXT Mistral uses [INST] ... [/INST] format
        image_tokens = "\n".join(["<image>"] * num)
        prompt = f"[INST] {image_tokens}\n{SYSTEM_PROMPT}\n\n{user_prompt} [/INST]"

        start = time.time()
        inputs = self.processor(
            text=prompt, images=images, return_tensors="pt", padding=True
        ).to(self.model.device)
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=150, do_sample=False
            )
        latency = time.time() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return caption, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()


# --- Adapter 2: Qwen2-VL-7B-Instruct ---

class Qwen2VLAdapter(BaseVLMAdapter):

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id}")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=qconfig,
            device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = build_user_prompt(num)

        image_content = [{"type": "image", "image": img} for img in images]
        user_content = image_content + [{"type": "text", "text": user_prompt}]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.time()
        inputs = self.processor(
            text=[text_input], images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=150, do_sample=False
            )
        latency = time.time() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return caption, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()


# --- Adapter 3: InternVL2-8B ---

class InternVL2Adapter(BaseVLMAdapter):

    def load(self):
        from transformers import AutoModel, AutoTokenizer
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        print(f"  [LOAD] {self.model_id}")

        # InternVL2 + 4-bit quantization has known meta tensor issues.
        # Load in bfloat16 directly — uses more VRAM but always works.
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = build_user_prompt(num)

        pixel_values_list = [self.transform(img).unsqueeze(0) for img in images]
        pixel_values = torch.cat(pixel_values_list, dim=0).to(
            torch.bfloat16
        ).to(self.model.device)
        num_patches_list = [1] * num

        image_placeholders = "".join(
            [f"Image-{i+1}: <image>\n" for i in range(num)]
        )
        question = f"{image_placeholders}{SYSTEM_PROMPT}\n\n{user_prompt}"

        generation_config = dict(max_new_tokens=150, do_sample=False)

        start = time.time()
        try:
            caption = self.model.chat(
                self.tokenizer, pixel_values, question,
                generation_config, num_patches_list=num_patches_list
            )
        except Exception as e:
            print(f"  [WARN] InternVL2 chat failed: {e}")
            caption = f"[ERROR] {str(e)[:100]}"
        latency = time.time() - start

        return caption.strip(), latency

    def unload(self):
        del self.model, self.tokenizer, self.transform
        self._cleanup_gpu()


# --- Adapter 4: Phi-3.5-Vision-Instruct ---

class Phi35VisionAdapter(BaseVLMAdapter):

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id}")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            num_crops=4,
        )

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = build_user_prompt(num)

        # Phi-3.5-Vision uses <|image_N|> placeholders (1-indexed)
        image_placeholders = "".join(
            [f"<|image_{i+1}|>\n" for i in range(num)]
        )

        messages = [
            {"role": "user", "content": f"{image_placeholders}{SYSTEM_PROMPT}\n\n{user_prompt}"},
        ]

        text_input = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start = time.time()
        inputs = self.processor(
            text=text_input, images=images, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                use_cache=False,  # Fix: disable DynamicCache to avoid seen_tokens error
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
        latency = time.time() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return caption, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()


# --- Adapter 5: MiniCPM-V-4.5 ---

class MiniCPMV45Adapter(BaseVLMAdapter):

    def load(self):
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel, AutoProcessor
        print(f"  [LOAD] {self.model_id}")

        # Fix: Patch for transformers versions missing _tied_weights_keys
        if not hasattr(PreTrainedModel, '_tied_weights_keys'):
            PreTrainedModel._tied_weights_keys = []
        if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
            PreTrainedModel.all_tied_weights_keys = property(
                lambda self: getattr(self, '_tied_weights_keys', [])
            )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = f"{SYSTEM_PROMPT}\n\n{build_user_prompt(num)}"

        msgs = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        start = time.time()
        try:
            caption = self.model.chat(
                tokenizer=self.tokenizer,
                msgs=msgs,
                images=images,
                max_new_tokens=150
            )
        except Exception as e:
            print(f"  [WARN] MiniCPM-V-4.5 chat failed: {e}")
            caption = f"[ERROR] {str(e)[:100]}"
        latency = time.time() - start

        return caption.strip(), latency

    def unload(self):
        for attr in ["model", "tokenizer", "processor"]:
            if hasattr(self, attr):
                delattr(self, attr)

        self._cleanup_gpu()

class Qwen25VLImageAdapter(BaseVLMAdapter):
    """Qwen2.5-VL (3B or 7B) in IMAGE mode."""
    def __init__(self, mid):
        super().__init__(mid); self.input_mode = "image"
    def load(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id} 🖼️ IMAGE")
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=q, device_map="auto", trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    def generate_caption(self, frame_paths=None, **kw):
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        n = len(images)
        ic = [{"type": "image", "image": img} for img in images]
        uc = ic + [{"type": "text", "text": build_user_prompt(n)}]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": uc}]
        ti = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = time.time()
        inp = self.processor(text=[ti], images=images, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=150, do_sample=False)
        lat = time.time() - t
        tr = [o[len(i):] for i, o in zip(inp.input_ids, g)]
        cap = self.processor.batch_decode(tr, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return cap, lat
    def unload(self):
        del self.model, self.processor; self._cleanup_gpu()


# --- Adapter 6: LLaVA-OneVision-7B ---

class LLaVAOneVisionAdapter(BaseVLMAdapter):

    def load(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
        print(f"  [LOAD] {self.model_id}")
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=qconfig,
            device_map="auto", trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    def generate_caption(self, frame_paths: List[str]) -> Tuple[str, float]:
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        num = len(images)
        user_prompt = build_user_prompt(num)

        image_content = [{"type": "image"} for _ in images]
        user_content = image_content + [
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}
        ]

        messages = [{"role": "user", "content": user_content}]

        text_input = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        start = time.time()
        inputs = self.processor(
            text=text_input, images=images, return_tensors="pt"
        ).to(self.model.device, dtype=torch.float16)
        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=150, do_sample=False
            )
        latency = time.time() - start

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        caption = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return caption, latency

    def unload(self):
        del self.model, self.processor
        self._cleanup_gpu()


# --- Adapter Registry ---
ADAPTER_CLASSES = {
    "LLaVANextAdapter": LLaVANextAdapter,
    "Qwen2VLAdapter": Qwen2VLAdapter,
    "InternVL2Adapter": InternVL2Adapter,
    "Phi35VisionAdapter": Phi35VisionAdapter,
    "MiniCPMV45Adapter": MiniCPMV45Adapter,
    "LLaVAOneVisionAdapter": LLaVAOneVisionAdapter,
    "Qwen25VLImageAdapter": Qwen25VLImageAdapter,
}


# =============================================================================
# MAIN BENCHMARK PIPELINE
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)

    print("=" * 80)
    print("  AgentVigil — VLM Benchmarking Pipeline (Adaptive)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models:  {len(MODELS_TO_BENCHMARK)}")
    print(f"  Video:   {os.path.basename(VIDEO_INPUT_PATH)}")
    print("=" * 80)

    # ---- Step 1: Extract chunks and frames (ADAPTIVE) ----
    print("\n[STEP 1/5] Extracting video chunks and frames (adaptive)...")
    chunks = extract_chunks_and_frames(
        video_path=VIDEO_INPUT_PATH,
        output_dir=DEBUG_FRAMES_DIR,
        chunk_duration_sec=None,
        max_chunks=MAX_CHUNKS_FOR_BENCHMARK,
        sampling_strategy="uniform",
    )
    if not chunks:
        print("[FATAL] No chunks extracted. Exiting.")
        return

    # Determine the chunk duration that was auto-selected
    first_chunk = next(iter(chunks.values()))
    effective_chunk_duration = first_chunk["chunk_duration_sec"]

    # ---- Step 2: Load ground truth ----
    print("\n[STEP 2/5] Loading ground truth annotations...")
    ground_truth_map = load_ground_truth(
        UCA_ANNOTATIONS_JSON, VIDEO_INPUT_PATH, effective_chunk_duration
    )

    # ---- Step 3: Initialize metrics ----
    print("\n[STEP 3/5] Initializing metrics engine (BLEU, ROUGE, METEOR, Keyword)...")
    metrics_engine = MetricsEngine()

    # ---- Step 4: Run benchmarks ----
    print("\n[STEP 4/5] Running benchmarks...\n")
    detailed_rows = []
    summary_rows = []

    for model_name, (adapter_class_name, model_id) in MODELS_TO_BENCHMARK.items():
        print(f"\n{'─' * 70}")
        print(f"  Benchmarking: {model_name} ({model_id})")
        print(f"{'─' * 70}")

        AdapterClass = ADAPTER_CLASSES.get(adapter_class_name)
        if AdapterClass is None:
            print(f"  [ERROR] Unknown adapter: {adapter_class_name}. Skipping.")
            continue

        adapter = AdapterClass(model_id)

        try:
            adapter.load()
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_name}: {e}")
            adapter._cleanup_gpu()
            continue

        model_latencies = []
        model_metrics = {
            "BLEU-1": [], "BLEU-2": [], "ROUGE-1 F1": [], "ROUGE-2 F1": [],
            "ROUGE-L F1": [], "METEOR": [], "Keyword Hit Rate": [],
        }
        chunks_with_gt = 0
        chunks_processed = 0
        total_model_start = time.time()

        for chunk_idx, chunk_info in sorted(chunks.items()):
            frame_paths = chunk_info["frame_paths"]

            try:
                caption, latency = adapter.generate_caption(frame_paths)
            except Exception as e:
                print(f"  [ERROR] Chunk {chunk_idx} failed: {e}")
                caption, latency = f"[ERROR] {str(e)[:80]}", 0.0

            model_latencies.append(latency)
            chunks_processed += 1

            # Check for ground truth
            gt_text = ground_truth_map.get(chunk_idx, "")
            has_gt = bool(gt_text)

            # Compute metrics if ground truth exists
            metrics_dict = {}
            if has_gt:
                metrics_dict = metrics_engine.compute_all(caption, gt_text)
                chunks_with_gt += 1
                for k in model_metrics:
                    if k in metrics_dict:
                        model_metrics[k].append(metrics_dict[k])

            # Build detailed CSV row
            row = {
                "Model": model_name,
                "Model ID": model_id,
                "Chunk Index": chunk_idx,
                "Chunk Start (s)": chunk_info["start_sec"],
                "Chunk End (s)": chunk_info["end_sec"],
                "Chunk Duration (s)": chunk_info["chunk_duration_sec"],
                "Num Frames": chunk_info["num_frames"],
                "Is Sampled": chunk_info["is_sampled"],
                "Frame Paths": "; ".join(
                    [os.path.basename(p) for p in frame_paths]
                ),
                "Generated Caption": caption,
                "Ground Truth": gt_text if has_gt else "N/A",
                "Has Ground Truth": has_gt,
                "Latency (s)": round(latency, 3),
                "BLEU-1": metrics_dict.get("BLEU-1", ""),
                "BLEU-2": metrics_dict.get("BLEU-2", ""),
                "ROUGE-1 F1": metrics_dict.get("ROUGE-1 F1", ""),
                "ROUGE-2 F1": metrics_dict.get("ROUGE-2 F1", ""),
                "ROUGE-L F1": metrics_dict.get("ROUGE-L F1", ""),
                "METEOR": metrics_dict.get("METEOR", ""),
                "Keyword Hits": metrics_dict.get("Keyword Hits", ""),
                "Keyword Total (GT)": metrics_dict.get("Keyword Total (GT)", ""),
                "Keyword Hit Rate": metrics_dict.get("Keyword Hit Rate", ""),
            }
            detailed_rows.append(row)

            status = (f"BLEU-2={metrics_dict.get('BLEU-2', 'N/A')}"
                      if has_gt else "no GT")
            print(f"  Chunk {chunk_idx:3d} | "
                  f"{chunk_info['start_sec']:6.1f}s - {chunk_info['end_sec']:6.1f}s | "
                  f"Frames: {chunk_info['num_frames']} | "
                  f"Latency: {latency:.2f}s | {status}")
            print(f"           Caption: {caption[:120]}...")

        total_model_time = time.time() - total_model_start

        # Build summary row
        avg_latency = (sum(model_latencies) / len(model_latencies)
                       if model_latencies else 0)
        summary = {
            "Model": model_name,
            "Model ID": model_id,
            "Total Chunks": chunks_processed,
            "Chunks with GT": chunks_with_gt,
            "Chunk Duration (s)": effective_chunk_duration,
            "Total Time (s)": round(total_model_time, 2),
            "Avg Latency (s)": round(avg_latency, 3),
            "Min Latency (s)": round(min(model_latencies), 3) if model_latencies else 0,
            "Max Latency (s)": round(max(model_latencies), 3) if model_latencies else 0,
            "Throughput (chunks/min)": round(
                (chunks_processed / total_model_time) * 60, 1
            ) if total_model_time > 0 else 0,
        }
        for metric_name, values in model_metrics.items():
            if values:
                summary[f"Avg {metric_name}"] = round(
                    sum(values) / len(values), 4
                )
            else:
                summary[f"Avg {metric_name}"] = "N/A"

        summary_rows.append(summary)

        print(f"\n  ✓ {model_name} complete: {chunks_processed} chunks "
              f"in {total_model_time:.1f}s "
              f"({avg_latency:.2f}s avg/chunk)")

        adapter.unload()

    # ---- Step 5: Export results ----
    print(f"\n{'=' * 80}")
    print("[STEP 5/5] Exporting results...")

    # Detailed CSV
    detailed_csv_path = os.path.join(
        OUTPUT_DIR, f"benchmark_detailed_{timestamp}.csv"
    )
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"  ✓ Detailed CSV ({len(detailed_rows)} rows): {detailed_csv_path}")

    # Summary CSV
    summary_csv_path = os.path.join(
        OUTPUT_DIR, f"benchmark_summary_{timestamp}.csv"
    )
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"  ✓ Summary CSV  ({len(summary_rows)} rows):  {summary_csv_path}")

    # Print summary table to console
    print(f"\n{'=' * 80}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 80}\n")
    print(df_summary.to_string(index=False))

    # Print comparison table (key metrics side-by-side)
    if len(summary_rows) > 1:
        print(f"\n{'─' * 70}")
        print("  MODEL COMPARISON (Key Metrics)")
        print(f"{'─' * 70}")
        comparison_cols = [
            "Model", "Avg Latency (s)", "Throughput (chunks/min)",
            "Avg BLEU-2", "Avg ROUGE-L F1", "Avg METEOR",
            "Avg Keyword Hit Rate",
        ]
        available_cols = [c for c in comparison_cols if c in df_summary.columns]
        print(df_summary[available_cols].to_string(index=False))

    print(f"\n{'=' * 80}")
    print(f"  Benchmark complete!")
    print(f"  Detailed: {detailed_csv_path}")
    print(f"  Summary:  {summary_csv_path}")
    print(f"  Frames:   {DEBUG_FRAMES_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
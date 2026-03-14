"""
Video Processor — Stage 0
===========================
WHAT:  Video file → adaptive chunks → extracted frame JPEGs

REPLICATED FROM: benchmark_vlm.py (the exact code that produced our
winning results across 4 crime categories)

KEY FUNCTIONS COPIED FROM BENCHMARK:
  - determine_chunk_duration()  → identical tier logic
  - select_chunk_indices()      → identical sampling strategies
  - extract_chunks_and_frames() → identical extraction with same naming

OPTIONAL ADDITION:
  - Sliding window overlap (set CHUNK_OVERLAP_SEC > 0 in config.py)
  - Default is 0.0 = exact benchmark behavior (hard cuts, no overlap)

NAMING CONVENTION:
  Benchmark: {video}_chunk{0001}_frame{0}.jpg
  This file: {video}_chunk{0001}_frame{0}.jpg  ← IDENTICAL
  (Important for ground truth alignment)
"""

import os
import cv2
from typing import Dict, List, Any

from config import (
    CHUNK_DURATION_TIERS,
    MAX_CHUNKS,
    FRAMES_PER_SECOND,
    SAMPLING_STRATEGY,
    CHUNK_OVERLAP_SEC,
)


def determine_chunk_duration(total_duration_sec: float) -> int:
    """
    Selects optimal chunk duration based on video length.
    IDENTICAL to benchmark_vlm.py logic.
    """
    for threshold, duration in CHUNK_DURATION_TIERS:
        if total_duration_sec <= threshold:
            return duration
    return CHUNK_DURATION_TIERS[-1][1]


def select_chunk_indices(
    total_chunks: int,
    max_chunks: int,
    strategy: str = "uniform",
) -> List[int]:
    """
    When total chunks exceed budget, selects a representative subset.
    IDENTICAL to benchmark_vlm.py — supports all 3 strategies.

    Strategies:
        "uniform":  Evenly spaced across video (default, benchmark-proven)
        "edges":    Prioritize start + end (events at boundaries)
        "all":      No sampling, return everything
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
            middle_indices = [
                int(middle_start + i * middle_step)
                for i in range(middle_count)
            ]
        else:
            middle_indices = []
        return sorted(set(start_indices + middle_indices + end_indices))

    return list(range(min(total_chunks, max_chunks)))


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata for UI display."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 1),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = round(
        info["total_frames"] / max(info["fps"], 1), 1
    )
    info["chunk_duration"] = determine_chunk_duration(info["duration_sec"])
    info["overlap_sec"] = CHUNK_OVERLAP_SEC
    info["sampling_strategy"] = SAMPLING_STRATEGY

    # Calculate expected chunk count
    if CHUNK_OVERLAP_SEC > 0:
        stride = max(info["chunk_duration"] - CHUNK_OVERLAP_SEC, 1.0)
        info["stride_sec"] = stride
        info["total_chunks"] = max(
            1,
            int((info["duration_sec"] - info["chunk_duration"]) / stride) + 1,
        )
    else:
        info["stride_sec"] = info["chunk_duration"]
        total_possible = max(
            1, int(info["duration_sec"] / info["chunk_duration"])
        )
        if info["duration_sec"] % info["chunk_duration"] >= 0.5:
            total_possible += 1
        info["total_chunks"] = min(total_possible, MAX_CHUNKS)

    cap.release()
    return info


def extract_chunks_and_frames(
    video_path: str,
    output_dir: str,
    chunk_duration_sec: int = None,
    max_chunks: int = MAX_CHUNKS,
    sampling_strategy: str = SAMPLING_STRATEGY,
    overlap_sec: float = CHUNK_OVERLAP_SEC,
) -> Dict[int, Dict[str, Any]]:
    """
    Adaptively splits video into chunks and extracts 1 frame/sec.

    When overlap_sec = 0.0 (default):
        EXACT replication of benchmark_vlm.py behavior.
        Hard cuts, no overlap, same naming convention.

    When overlap_sec > 0.0:
        Sliding window mode. Each chunk overlaps with the next.
        Catches events at chunk boundaries.

    Args:
        video_path:          Path to input video
        output_dir:          Where to save frame JPEGs
        chunk_duration_sec:  Override chunk size (None = auto from tiers)
        max_chunks:          Maximum chunks to process
        sampling_strategy:   "uniform" | "edges" | "all"
        overlap_sec:         Overlap between chunks (0.0 = benchmark mode)

    Returns:
        {chunk_index: {
            "frame_paths": [str, ...],
            "start_sec": float,
            "end_sec": float,
            "num_frames": int,
            "chunk_duration_sec": int,
            "is_sampled": bool,
            "overlap_sec": float,
            "stride_sec": float,
        }}
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)

    # ── Open video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_path}")

    total_duration = total_frames / fps

    # ── Determine chunk duration (adaptive) ─────────────────────────
    if chunk_duration_sec is None:
        chunk_duration_sec = determine_chunk_duration(total_duration)

    # ── Compute stride and generate chunk start positions ───────────
    overlap_sec = min(overlap_sec, chunk_duration_sec - 0.5)
    overlap_sec = max(overlap_sec, 0.0)
    stride_sec = chunk_duration_sec - overlap_sec

    if overlap_sec > 0:
        # Sliding window mode
        chunk_starts = []
        pos = 0.0
        while pos < total_duration:
            chunk_end = pos + chunk_duration_sec
            if chunk_end > total_duration + 0.5:
                if total_duration - pos >= 1.0:
                    chunk_starts.append(pos)
                break
            chunk_starts.append(pos)
            pos += stride_sec

        total_possible_chunks = len(chunk_starts)
    else:
        # Benchmark mode: hard cuts, identical to benchmark_vlm.py
        total_possible_chunks = max(
            1, int(total_duration / chunk_duration_sec)
        )
        if total_duration % chunk_duration_sec >= 0.5:
            total_possible_chunks += 1

        chunk_starts = [
            i * chunk_duration_sec for i in range(total_possible_chunks)
        ]

    # ── Sample chunks if too many ───────────────────────────────────
    is_sampled = total_possible_chunks > max_chunks
    selected_indices = select_chunk_indices(
        total_possible_chunks, max_chunks, sampling_strategy
    )
    selected_set = set(selected_indices)

    mode_label = f"overlap={overlap_sec}s, stride={stride_sec}s" if overlap_sec > 0 else "hard cuts (benchmark mode)"

    print(f"[VIDEO] {video_name}")
    print(f"  FPS: {fps:.1f} | Duration: {total_duration:.1f}s | "
          f"Total Frames: {total_frames}")
    print(f"  Adaptive Chunk Duration: {chunk_duration_sec}s "
          f"(auto-selected for {total_duration:.0f}s video)")
    print(f"  Mode: {mode_label}")
    print(f"  Total Possible Chunks: {total_possible_chunks}")
    print(f"  Chunks Selected: {len(selected_indices)}"
          f"{' (SAMPLED)' if is_sampled else ' (all)'}"
          f" | Strategy: {sampling_strategy}")

    # ── Extract frames chunk by chunk ───────────────────────────────
    chunks = {}

    for chunk_index, pos_index in enumerate(selected_indices):
        if pos_index >= len(chunk_starts):
            break

        current_start = chunk_starts[pos_index]
        chunk_end = min(current_start + chunk_duration_sec, total_duration)
        actual_duration = chunk_end - current_start

        if actual_duration < 0.5:
            continue

        # Extract 1 frame per second within this chunk
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

            # NAMING: identical to benchmark_vlm.py
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
                "overlap_sec": overlap_sec,
                "stride_sec": stride_sec,
            }

    cap.release()

    total_extracted = sum(c["num_frames"] for c in chunks.values())
    print(f"  Result: {len(chunks)} chunks | {total_extracted} total frames "
          f"| Saved to: {frames_dir}")

    return chunks
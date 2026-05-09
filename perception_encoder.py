"""
perception_encoder.py — Runtime PE wrapper for Flashback gate
================================================================
WHAT:  Frozen video+text encoder used at INFERENCE TIME by
       flashback_filter.py. Encodes chunk frames into 1024-D vectors
       and matches them against the pre-built pseudo_scene_memory bank.

API IMPORT PATH (CRITICAL)
---------------------------
Meta's perception_models repo is installed as an editable package.
The actual import path is:
    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

NOT `import perception_models as pm` (that doesn't exist on PyPI).
NOT `from transformers import AutoModel` (no HF mirror of PE).

Install once with:
    git clone https://github.com/facebookresearch/perception_models.git
    cd perception_models
    pip install -e .

Verify with:
    python -c "import core.vision_encoder.pe as pe; print(pe.CLIP.available_configs())"

CONNECTS TO
-----------
    flashback_filter.py    — calls encode_video(frame_paths) per chunk
    pseudo_scene_memory.py — provides matched bank built offline
    encode_pairs.py        — uses the SAME backbone for memory build,
                             so embed_dim must match between the two
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image


# ── Defaults (must match config.py / encode_pairs.py) ───────────────
DEFAULT_PE_MODEL = "PE-Core-L14-336"      # 1024-D, fits with VLM on 12 GB
DEFAULT_FRAMES_PER_SEGMENT = 16            # paper's Tsample = 16 frames


class PerceptionEncoder:
    """
    Frozen Meta Perception Encoder for video + text encoding.

    PUBLIC API
    ----------
        load()                              — call once at app startup
        encode_video(frame_paths) -> ndarray (D,)    L2-normalised
        encode_texts(list[str])   -> ndarray (N, D)  L2-normalised
        unload()                            — release GPU

    Both encoders return float32 numpy arrays. Vectors are unit-norm,
    so dot product == cosine similarity downstream in flashback_filter.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PE_MODEL,
        frames_per_segment: int = DEFAULT_FRAMES_PER_SEGMENT,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.frames_per_segment = frames_per_segment
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.preprocess = None     # image transform
        self.tokenizer = None      # text tokenizer
        self.embed_dim: Optional[int] = None
        self.image_size: Optional[int] = None
        self.loaded = False

    # ── LOAD ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load PE weights via the official Meta repo API."""
        if self.loaded:
            return

        try:
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms
        except ImportError as e:
            raise RuntimeError(
                f"Could not import core.vision_encoder.pe ({e}).\n"
                "  perception_models is not installed correctly. Fix:\n"
                "    git clone https://github.com/facebookresearch/perception_models.git\n"
                "    cd perception_models\n"
                "    pip install -e .\n"
                "  Verify: python -c 'import core.vision_encoder.pe as pe; print(\"OK\")'"
            )

        # Sanity check on requested config
        avail = pe.CLIP.available_configs()
        if self.model_name not in avail:
            raise ValueError(
                f"Unknown PE config '{self.model_name}'. Available: {avail}"
            )

        print(f"[PE] Loading {self.model_name} on {self.device}...")
        # Factory returns CPU model; move to device manually.
        self.model = pe.CLIP.from_config(self.model_name, pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image + text preprocessing factories live in `transforms`,
        # NOT `pe`. They read sizing parameters off the loaded model.
        self.image_size = self.model.image_size
        self.preprocess = transforms.get_image_transform(self.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

        for p in self.model.parameters():
            p.requires_grad_(False)

        # Discover output dim with one dummy text forward
        with torch.no_grad():
            dummy_tokens = self.tokenizer(["test"]).to(self.device)
            feats = self.model.encode_text(dummy_tokens)
            self.embed_dim = int(feats.shape[-1])

        self.loaded = True
        print(f"[PE] Ready — output dim {self.embed_dim}, "
              f"image_size {self.image_size}, "
              f"{self.frames_per_segment} frames/segment")

    # ── VIDEO ENCODE ────────────────────────────────────────────────

    def encode_video(self, frame_paths: List[str]) -> np.ndarray:
        """
        Encode a chunk into a single L2-normalised feature vector.

        PE-Core is trained as an image+text encoder with a video
        extension: we encode each frame independently then mean-pool.
        This matches the official Colab demo.
        """
        if not self.loaded:
            self.load()
        if not frame_paths:
            raise ValueError("encode_video: empty frame_paths")

        frames = self._sample_and_load(frame_paths)   # (T, 3, H, W) on device

        with torch.no_grad():
            # Per-frame encoding
            frame_feats = self.model.encode_image(frames)   # (T, D)
            # L2-normalise per frame, then mean-pool, then re-normalise.
            # This is the standard CLIP-video recipe.
            frame_feats = torch.nn.functional.normalize(frame_feats, dim=-1)
            video_feats = frame_feats.mean(dim=0, keepdim=True)
            video_feats = torch.nn.functional.normalize(video_feats, dim=-1)

        return video_feats.squeeze(0).cpu().numpy().astype(np.float32)

    def _sample_and_load(self, frame_paths: List[str]) -> torch.Tensor:
        """Uniformly sample frames_per_segment frames, load + preprocess."""
        n = len(frame_paths)
        if n >= self.frames_per_segment:
            idx = np.linspace(0, n - 1, self.frames_per_segment, dtype=int)
        else:
            # Pad by repeating the last frame — fixes the temporal axis.
            idx = list(range(n)) + [n - 1] * (self.frames_per_segment - n)

        tensors = []
        for i in idx:
            img = Image.open(frame_paths[i]).convert("RGB")
            tensors.append(self.preprocess(img))
        return torch.stack(tensors).to(self.device)

    # ── TEXT ENCODE ─────────────────────────────────────────────────

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Encode a list of strings into L2-normalised text features."""
        if not self.loaded:
            self.load()
        if not texts:
            return np.zeros((0, self.embed_dim or 1), dtype=np.float32)

        all_feats = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy().astype(np.float32))

        return np.concatenate(all_feats, axis=0)

    # ── UNLOAD ──────────────────────────────────────────────────────

    def unload(self) -> None:
        if self.model is not None:
            del self.model, self.preprocess, self.tokenizer
            self.model = self.preprocess = self.tokenizer = None
        self.loaded = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
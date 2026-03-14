"""
Embedding Engine
=================
WHAT:  Converts text captions → 384-dimensional numeric vectors.

WHY:   Milvus (Vector Index in Stage 1) stores these vectors and
       finds "similar" ones via cosine distance. This enables:
       - "Find clips that look like this robbery"
       - "Is this scene similar to past incidents?"
       Without embeddings, we can only do exact keyword matching.

HOW:   Uses sentence-transformers/all-MiniLM-L6-v2
       - 384 dimensions (compact but expressive)
       - ~0.05s per caption (negligible latency)
       - Captures semantic meaning, not just keywords

CONNECTS TO: database.py stores these vectors in Milvus
             agent.py uses these to search similar incidents
"""

from sentence_transformers import SentenceTransformer
from typing import List
from config import EMBEDDING_MODEL, EMBEDDING_DIM


class EmbeddingEngine:
    def __init__(self):
        print(f"[EMBED] Loading {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[EMBED] Ready ({EMBEDDING_DIM}-dim vectors)")

    def embed_text(self, text: str) -> List[float]:
        """Single caption → vector."""
        return self.model.encode(text).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Multiple captions → vectors (batched = faster)."""
        return self.model.encode(texts).tolist()

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM
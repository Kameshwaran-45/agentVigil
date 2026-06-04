from adapters.base import BaseVideoAdapter
from adapters.llava_adapter import LLaVAAdapter
from adapters.videollama3_adapter import VideoLLaMA3Adapter
from .qwen25vl_adapter import Qwen25VLAdapter

ADAPTER_CLASSES = {
    "VideoLLaMA3Adapter": VideoLLaMA3Adapter,
    "LLaVAAdapter":       LLaVAAdapter,
    "Qwen25VLAdapter":    Qwen25VLAdapter,  # ← new
}

__all__ = [
    "BaseVideoAdapter",
    "LLaVAAdapter",
    "VideoLLaMA3Adapter",
    "ADAPTER_CLASSES",
]
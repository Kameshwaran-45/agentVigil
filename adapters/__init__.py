from adapters.base import BaseVideoAdapter
from adapters.llava_adapter import LLaVAAdapter
from adapters.videollama3_adapter import VideoLLaMA3Adapter

ADAPTER_CLASSES = {
    "LLaVAAdapter": LLaVAAdapter,
    "VideoLLaMA3Adapter": VideoLLaMA3Adapter,
}

__all__ = [
    "BaseVideoAdapter",
    "LLaVAAdapter",
    "VideoLLaMA3Adapter",
    "ADAPTER_CLASSES",
]

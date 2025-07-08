"""
Speech-to-Text (STT) processors for real-time audio transcription.

This module provides:
- BaseSTTProcessor: Abstract base class for STT implementations
- Factory function to create different STT processors
"""

import tempfile
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

logger = logging.getLogger(__name__)

class BaseSTTProcessor(ABC):
    """Abstract base class for STT processors"""
    
    @abstractmethod
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe audio file bytes to text"""
        pass
    
    @abstractmethod
    async def process_audio_chunk(self, audio_chunk) -> Tuple[str, float]:
        """Process real-time audio chunk and return (text, processing_time_ms)"""
        pass
    
    @abstractmethod
    def reset_state(self):
        """Reset processor state between sessions"""
        pass

# ================== STT FACTORY ==================
def create_stt_processor(processor_type: str = "nemo", **kwargs) -> BaseSTTProcessor:
    """Factory function to create the selected STT processor
    
    Args:
        processor_type: "nemo" or "faster-whisper"
        **kwargs: Additional arguments for specific processors
    """
    if processor_type == "nemo":
        from .nemo_stt import NeMoSTTProcessor
        return NeMoSTTProcessor()
    elif processor_type == "faster-whisper":
        from .faster_whisper_stt import FasterWhisperSTTProcessor
        return FasterWhisperSTTProcessor(**kwargs)
    else:
        raise ValueError(f"Unknown processor type: {processor_type}. Supported: 'nemo', 'faster-whisper'")

# Legacy compatibility - keep the old factory function
def create_stt_processor_legacy(model_name: str = "realtime") -> BaseSTTProcessor:
    """Legacy factory function for backward compatibility"""
    if model_name == "realtime":
        from .nemo_stt import NeMoSTTProcessor
        return NeMoSTTProcessor()
    else:
        raise ValueError(f"Unknown STT model: {model_name}. Only 'realtime' is supported.")
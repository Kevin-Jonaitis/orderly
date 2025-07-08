"""
Faster Whisper STT Processor
Real-time STT using Faster Whisper with CTranslate2 for optimized performance
"""

import tempfile
import time
import logging
import numpy as np
import os
from datetime import datetime
from typing import Tuple, Optional
import io
import wave

from .stt import BaseSTTProcessor

logger = logging.getLogger(__name__)

class FasterWhisperSTTProcessor(BaseSTTProcessor):
    """Real-time STT using Faster Whisper with streaming support"""
    
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str = "float16"):
        """
        Initialize Faster Whisper STT processor
        
        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3")
            device: Device to run on ("cuda", "cpu")
            compute_type: Compute type ("float16", "int8_float16", "int8")
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        logger.info(f"🚀 Loading Faster Whisper model: {model_size} on {device} with {compute_type}")
        
        try:
            from faster_whisper import WhisperModel
            
            # Load model with specified parameters
            self.model = WhisperModel(
                model_size_or_path=model_size,
                device=device,
                compute_type=compute_type
            )
            
                    # Initialize state
        self.reset_state()
        
        # Audio configuration
        self.sample_rate = 16000  # Faster Whisper expects 16kHz
            
            logger.info(f"✅ Faster Whisper model loaded successfully")
            logger.info(f"📊 Audio: {self.sample_rate}Hz")
            
        except ImportError as e:
            logger.error("❌ Faster Whisper not installed. Install with: pip install faster-whisper")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load Faster Whisper model: {e}")
            raise e
    
    def reset_state(self):
        """Reset state between sessions"""
        self.step_num = 0
        
    def _audio_to_wav_bytes(self, audio_chunk: np.ndarray) -> bytes:
        """Convert audio chunk to WAV bytes for Faster Whisper"""
        # Ensure audio is in the right format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Convert from float32 [-1, 1] to int16 [-32768, 32767]
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue()
    
    async def transcribe(self, wav_bytes: bytes) -> str:
        """File-based transcription for compatibility with BaseSTTProcessor"""
        try:
            # Save wav_bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(wav_bytes)
                tmp_file.flush()
                audio_file = tmp_file.name
            
            if not os.path.exists(audio_file):
                logger.error(f"❌ Audio file not found: {audio_file}")
                return ""
            
            # Transcribe with Faster Whisper
            start_time = time.time()
            segments, info = self.model.transcribe(
                audio_file,
                beam_size=5,
                language="en",
                condition_on_previous_text=False
            )
            
            # Collect all segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + " "
            
            process_time = (time.time() - start_time) * 1000
            
            logger.info(f"🎤 FASTER WHISPER STT: {process_time:.0f}ms → '{transcribed_text.strip()}'")
            
            # Cleanup
            try:
                os.unlink(audio_file)
            except:
                pass
            
            return transcribed_text.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in Faster Whisper transcription: {e}")
            return ""
    
 
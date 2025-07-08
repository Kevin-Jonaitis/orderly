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
            
            # Initialize streaming state
            self.reset_state()
            
            # Audio buffering for streaming
            self.audio_buffer = []
            self.sample_rate = 16000  # Faster Whisper expects 16kHz
            self.chunk_duration_ms = 80  # 80ms chunks
            self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)  # 1280 samples
            
            # Minimum buffer size for processing (similar to NeMo approach)
            self.min_chunks = 2  # 160ms minimum
            self.min_samples = self.chunk_size * self.min_chunks
            
            logger.info(f"✅ Faster Whisper model loaded successfully")
            logger.info(f"📊 Audio: {self.sample_rate}Hz, {self.chunk_duration_ms}ms chunks ({self.chunk_size} samples)")
            
        except ImportError as e:
            logger.error("❌ Faster Whisper not installed. Install with: pip install faster-whisper")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load Faster Whisper model: {e}")
            raise e
    
    def reset_state(self):
        """Reset streaming state between sessions"""
        self.step_num = 0
        self.audio_buffer = []
        
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
    
    async def process_audio_chunk(self, audio_chunk) -> Tuple[str, float]:
        """Process 80ms audio chunks for real-time streaming"""
        try:
            # Add to buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Process when we have enough chunks for stable processing
            if len(self.audio_buffer) < self.min_samples:
                # Need more chunks for stable processing
                return "", 0.0
            
            # Take exactly min_chunks worth of audio
            audio_to_process = np.array(self.audio_buffer[:self.min_samples])
            # Remove the processed audio, keep remainder
            self.audio_buffer = self.audio_buffer[self.chunk_size:]  # Remove 1 chunk, keep overlap
            
            # Convert to WAV bytes
            wav_bytes = self._audio_to_wav_bytes(audio_to_process)
            
            # Process with Faster Whisper
            start_time = time.time()
            
            # Use transcribe method with streaming-like approach
            segments, info = self.model.transcribe(
                io.BytesIO(wav_bytes),
                beam_size=5,
            )
            
            process_time = (time.time() - start_time) * 1000
            
            # Extract transcription from segments
            current_text = ""
            for segment in segments:
                current_text += segment.text + " "
            
            current_text = current_text.strip()
            
            # Log results
            if current_text:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🎤 FASTER WHISPER [{self.step_num:3d}] {process_time:3.0f}ms: '{current_text}'")
            elif self.step_num % 10 == 0:  # Show progress every 10 processed chunks 
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🔄 [{self.step_num:3d}] Processing...")
            
            self.step_num += 1
            return current_text, process_time
            
        except Exception as e:
            print(f"[Faster Whisper ERROR] Exception in process_audio_chunk: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0 
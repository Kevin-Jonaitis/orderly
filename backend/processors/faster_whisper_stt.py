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
from typing import Tuple, Optional, Dict, Any
import io
import wave
import soundfile as sf

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
        self.manual_speech_end_timestamp = None
        
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
        
    async def warm_up_stt_model(self):
        """Warm up the STT model by transcribing test audio"""
        print("🔥 [STT] Warming up STT model...")
        warmup_start_time = time.time()
        
        if os.path.exists("test_audio.wav"):
            try:
                print("🔥 [STT] Transcribing test_audio.wav for warm-up...")
                # Read test audio file
                audio_data, sample_rate = sf.read("test_audio.wav")
                # Convert to float32 if needed
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Convert to WAV bytes for simple transcription
                audio_array = np.array(audio_data, dtype=np.float32)
                wav_bytes = self._audio_to_wav_bytes(audio_array)
                
                # Call transcribe directly and await the result
                result = await self.transcribe(wav_bytes)
                warmup_time = (time.time() - warmup_start_time) * 1000
                print(f"✅ [STT] STT warm-up completed in {warmup_time:.1f}ms with result: '{result}'")
                    
            except Exception as e:
                print(f"❌ [STT] Failed to send STT warm-up audio: {e}")
        else:
            print("⚠️ [STT] test_audio.wav not found, skipping STT warm-up")
        
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
    
    async def process_audio_chunk(self, audio_chunk) -> Tuple[str, float]:
        """Process audio chunk - required by abstract base class but not used in silence-based approach"""
        # This method is required by the abstract base class but we don't use it
        # in the silence-based approach. Return empty result.
        return "", 0.0
    

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
    
    async def transcribe_buffer_with_timing(
        self, 
        audio_buffer: list
    ) -> str:
        """
        Transcribe audio buffer with detailed timing and statistics
        
        Args:
            audio_buffer: List of audio samples (already cleaned of trailing silence)
            
        Returns:
            Transcribed text string
        """
        if len(audio_buffer) == 0:
            return ""
        
        # Constants
        SAMPLE_RATE = 16000
        
        try:
            # Convert audio buffer to numpy array
            audio_array = np.array(audio_buffer, dtype=np.float32)
            
            # Convert to WAV bytes for STT processing with timing
            save_wav_start = time.time()
            wav_bytes = self._audio_to_wav_bytes(audio_array)
            save_wav_end = time.time()
            save_wav_time = (save_wav_end - save_wav_start) * 1000
            
            # Transcribe using the STT processor with timing
            transcribe_start = time.time()
            result = await self.transcribe(wav_bytes)
            transcribed_text = result.strip()
            transcribe_time = (time.time() - transcribe_start) * 1000
            
            # Record transcription complete time
            transcription_complete_time = time.time()
            
            # Calculate audio duration
            audio_duration = len(audio_buffer) / SAMPLE_RATE
            
            # Check if keyboard timing is available (manual_speech_end_timestamp > 0)
            keyboard_timing = None
            if self.manual_speech_end_timestamp and self.manual_speech_end_timestamp.value > 0:
                keyboard_timing = (transcription_complete_time - self.manual_speech_end_timestamp.value) * 1000
            
            # Print detailed stats if keyboard was pressed and we have transcribed text
            if (self.manual_speech_end_timestamp and self.manual_speech_end_timestamp.value > 0 and 
                transcribed_text):
                print(f"\n📊 STT TIMING STATS:")
                print(f"🎤 Text: '{transcribed_text}'")
                print(f" Save audio to WAV: {save_wav_time:.2f} ms")
                print(f"⏱️  Transcribe time: {transcribe_time:.2f} ms")
                print(f"🎵 Audio length: {audio_duration:.2f} seconds")
                print(f"📊 Buffer size: {len(audio_buffer)} samples")
                print(f"⌨️  Keyboard to transcription: {keyboard_timing:.2f}ms")
                print(f"🕒 Transcription complete time: {transcription_complete_time:.2f} s")
                print("-" * 50)
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"❌ Error in transcription: {e}")
            return ""
    
 
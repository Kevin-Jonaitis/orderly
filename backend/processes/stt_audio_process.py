"""
STT + WebRTC Audio Process
Processes audio from WebRTC connections and performs speech-to-text processing.
No longer captures from microphone - only processes WebRTC audio input.
"""

import sys
import os
import asyncio
import multiprocessing
import numpy as np
import queue
import time
import soundfile as sf
import io
import wave
import tempfile
import threading

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stt import create_stt_processor

# Audio configuration
SAMPLE_RATE = 16000  # Match frontend and NeMo STT sample rate
CHUNK_DURATION_MS = 20  # ms chunks for real-time processing

# Silence detection parameters
SILENCE_THRESHOLD = 0.005  # RMS threshold for silence detection
SILENCE_TIMEOUT_MS = 300  # 300ms of silence triggers transcription

def record_mic_buffer(mic_record_buffer, audio_data, last_record_time, sample_rate=16000, record_seconds=5, record_filename="debug_mic_input.wav"):
    """Accumulate audio data, keep last N seconds, and save to file every N seconds."""
    mic_record_buffer = np.concatenate([mic_record_buffer, audio_data])
    max_samples = sample_rate * record_seconds
    if len(mic_record_buffer) > max_samples:
        mic_record_buffer = mic_record_buffer[-max_samples:]
    now = time.time()
    if now - last_record_time > record_seconds:
        sf.write(record_filename, mic_record_buffer, sample_rate)
        last_record_time = now
    return mic_record_buffer, last_record_time

def detect_silence(audio_chunk):
    """Detect if audio chunk is silence based on RMS"""
    rms = np.sqrt(np.mean(audio_chunk**2))
    return rms < SILENCE_THRESHOLD

def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert audio array to WAV bytes for STT processing"""
    # Ensure audio is in the right format
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    
    # Convert from float32 [-1, 1] to int16 [-32768, 32767]
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return wav_buffer.getvalue()

class STTAudioProcess(multiprocessing.Process):
    """Process that handles WebRTC audio input and STT processing"""
    
    def __init__(self, text_queue, webrtc_audio_queue, last_text_change_timestamp, manual_speech_end_timestamp=None, stt_warmup_flag=None):
        super().__init__()
        self.text_queue = text_queue
        self.webrtc_audio_queue = webrtc_audio_queue  # Queue to receive WebRTC audio
        self.last_text_change_timestamp = last_text_change_timestamp
        self.manual_speech_end_timestamp = manual_speech_end_timestamp  # Shared timing variable
        self.stt_warmup_flag = stt_warmup_flag  # Shared flag to trigger STT warm-up
        
        # Silence detection variables
        self.audio_buffer = []
        self.silence_duration = 0
        self.segment_count = 0
        
        # Timing variables
        self.transcription_start_time: float | None = None
        self.transcription_complete_time: float | None = None
        self.silence_detected_time: float | None = None
        
        # Stats printing flag
        self.stats_printed: bool = False
        
    def run(self):
        """Main process loop - processes WebRTC audio and runs STT"""
        print("🎤 Starting STT+WebRTC Audio process...")
        
        # Initialize STT processor
        print("📝 Initializing STT processor...")
        stt_processor = create_stt_processor("faster-whisper", model_size="tiny", device="cuda", compute_type="int8_float16")
        # Set the manual speech end timestamp on the processor for timing measurements
        stt_processor.set_manual_speech_end_timestamp(self.manual_speech_end_timestamp)
        print("✅ STT processor loaded")
        
        print(" STT+WebRTC process started (using Faster Whisper with silence detection)")
        print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks")
        print(f" Silence threshold: {SILENCE_THRESHOLD}, timeout: {SILENCE_TIMEOUT_MS}ms")
        print(" Waiting for WebRTC audio input...")
        
        # Run async event loop to process WebRTC audio
        try:
            asyncio.run(self._process_webrtc_audio_loop(stt_processor))
        except Exception as e:
            print(f"❌ Error in WebRTC audio processing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _process_webrtc_audio_loop(self, stt_processor):
        """Async loop for processing WebRTC audio chunks with silence detection"""
        stt_warmed_up = False  # Track if STT has been warmed up
        chunk_count = 0
        
        while True:
            try:
                # Check if we need to warm up the STT model
                if (self.stt_warmup_flag and self.stt_warmup_flag.value == 1 and not stt_warmed_up):
                    print("🔥 [STT] Warming up STT model due to client connection...")
                    await stt_processor.warm_up_stt_model()
                    stt_warmed_up = True
                    # Reset the flag
                    self.stt_warmup_flag.value = 0
                    print("✅ [STT] STT model warmed up")
                
                # Get audio data from WebRTC queue with timing
                queue_get_start = time.time()
                audio_data = self.webrtc_audio_queue.get()
                queue_get_time = (time.time() - queue_get_start) * 1000
                
                if not isinstance(audio_data, np.ndarray):
                    audio_data = np.array(audio_data, dtype=np.float32)

                chunk_count += 1
                
                # Log queue timing every 100 chunks
                if chunk_count % 100 == 0:
                    print(f"📊 [STT] Chunk {chunk_count}: Queue get time: {queue_get_time:.2f}ms, Queue size: {self.webrtc_audio_queue.qsize()}")

                # Process the audio chunk with silence detection
                await self._process_audio_chunk(audio_data, stt_processor)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing WebRTC audio: {e}")
                continue
    
    async def _process_audio_chunk(self, audio_chunk, stt_processor):
        """Process a single audio chunk with silence detection"""
        try:
            # Add to audio buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Check for silence
            is_silence = detect_silence(audio_chunk)
            if is_silence:
                self.silence_duration += CHUNK_DURATION_MS
            else:
                self.silence_duration = 0  # Reset silence counter
            
            # Check if we should transcribe
            if self.silence_duration >= SILENCE_TIMEOUT_MS and len(self.audio_buffer) > 0:
                self.silence_detected_time = time.time()
                await self._transcribe_buffer(stt_processor)
                
        except Exception as e:
            print(f"❌ Error in STT processing: {e}")
    
    async def _transcribe_buffer(self, stt_processor):
        """Transcribe the current audio buffer using the STT processor's timing method"""
        if len(self.audio_buffer) == 0:
            return
        
        # Start timing for buffer-to-transcription
        buffer_transcribe_start = time.time()
        
        # Record transcription start time
        self.transcription_start_time = time.time()
        
        # Remove the last 300ms of silence from the buffer
        silence_samples = int(SILENCE_TIMEOUT_MS * SAMPLE_RATE / 1000)
        audio_without_silence = self.audio_buffer[:-silence_samples] if len(self.audio_buffer) > silence_samples else []
        
        # Check if we have enough audio to transcribe (at least 100ms)
        min_audio_samples = int(100 * SAMPLE_RATE / 1000)  # 100ms minimum
        if len(audio_without_silence) < min_audio_samples:
            # Reset buffer and silence counter
            self.audio_buffer = []
            self.silence_duration = 0
            return
        
        # Calculate audio duration
        audio_duration = len(self.audio_buffer) / SAMPLE_RATE
        
        # Calculate timing intervals
        silence_to_transcription = None
        if self.silence_detected_time is not None:
            silence_to_transcription = (self.transcription_start_time - self.silence_detected_time) * 1000
        
        # Use the STT processor's timing method with cleaned audio
        transcribed_text = await stt_processor.transcribe_buffer_with_timing(
            audio_buffer=audio_without_silence
        )
        
        # Record transcription complete time
        self.transcription_complete_time = time.time()
        
        # Calculate keyboard timing if available
        keyboard_timing = None
        keyboard_to_silence = None
        if self.manual_speech_end_timestamp and self.manual_speech_end_timestamp.value > 0:
            keyboard_timing = (self.transcription_complete_time - self.manual_speech_end_timestamp.value) * 1000
        if self.manual_speech_end_timestamp and self.manual_speech_end_timestamp.value > 0 and self.silence_detected_time is not None:
            keyboard_to_silence = (self.silence_detected_time - self.manual_speech_end_timestamp.value) * 1000
        
        if transcribed_text:
            # Update the last text change timestamp
            self.last_text_change_timestamp.value = time.time()
            
            # Send text to the queue
            try:
                self.text_queue.put(transcribed_text, block=False)
                print(f"📤 Sent transcribed text to LLM: '{transcribed_text}'")
            except Exception as e:
                print(f"❌ Failed to send text to LLM process: {e}")
            
            # Print detailed timing stats if keyboard was pressed
            if (self.manual_speech_end_timestamp and self.manual_speech_end_timestamp.value > 0 and 
                transcribed_text and not self.stats_printed):
                print(f"\n📊 STT PIPELINE TIMING BREAKDOWN:")
                print(f"🎤 Text: '{transcribed_text}'")
                print(f"🎵 Audio length: {audio_duration:.2f} seconds")
                print(f"📊 Buffer size: {len(self.audio_buffer)} samples")
                print(f"📊 Cleaned buffer size: {len(audio_without_silence)} samples")
                if silence_to_transcription is not None:
                    print(f"🔇 Silence detected → Transcription start: {silence_to_transcription:.2f}ms")
                if keyboard_to_silence is not None:
                    print(f"⌨️  Keyboard → Silence detected: {keyboard_to_silence:.2f}ms")
                if keyboard_timing is not None:
                    print(f"⌨️  Keyboard → Transcription complete: {keyboard_timing:.2f}ms")
                print(f"🕒 Transcription complete time: {self.transcription_complete_time:.3f} s")
                print("-" * 50)
                self.stats_printed = True
            
            # Print basic results (always)
            print(f"\n🎤 Segment: '{transcribed_text}'")
        
        # Reset buffer and silence counter
        self.audio_buffer = []
        self.silence_duration = 0
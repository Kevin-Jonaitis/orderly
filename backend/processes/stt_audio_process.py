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

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stt import create_stt_processor

# Audio configuration
SAMPLE_RATE = 16000  # Match frontend and NeMo STT sample rate
CHUNK_DURATION_MS = 20  # ms chunks for real-time processing

# Silence detection parameters
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
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

class STTAudioProcess(multiprocessing.Process):
    """Process that handles WebRTC audio input and STT processing"""
    
    def __init__(self, text_queue, webrtc_audio_queue, last_text_change_timestamp):
        super().__init__()
        self.text_queue = text_queue
        self.webrtc_audio_queue = webrtc_audio_queue  # Queue to receive WebRTC audio
        self.last_text_change_timestamp = last_text_change_timestamp
        
        # Silence detection variables
        self.audio_buffer = []
        self.silence_duration = 0
        self.segment_count = 0
        
    def run(self):
        """Main process loop - processes WebRTC audio and runs STT"""
        print("🎤 Starting STT+WebRTC Audio process...")
        
        # Initialize STT processor - let it crash if it fails
        print("📝 Initializing STT processor...")
        stt_processor = create_stt_processor("faster-whisper", model_size="tiny", device="cuda", compute_type="int8_float16")  # Use Faster Whisper for real-time processing
        print("✅ STT processor loaded")
        
        print("🎤 STT+WebRTC process started (using Faster Whisper with silence detection)")
        print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks")
        print(f"🔇 Silence threshold: {SILENCE_THRESHOLD}, timeout: {SILENCE_TIMEOUT_MS}ms")
        print("🔗 Waiting for WebRTC audio input...")
        
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
        mic_record_buffer = np.zeros(0, dtype=np.float32)
        last_record_time = time.time()
        RECORD_SECONDS = 5
        RECORD_FILENAME = "debug_mic_input.wav"
        
        while True:
            try:
                # Get audio data from WebRTC queue with timeout
                audio_data = self.webrtc_audio_queue.get()
                if not isinstance(audio_data, np.ndarray):
                    print("DATA NOT IN THE RIGHT FORMAT?")
                    audio_data = np.array(audio_data, dtype=np.float32)

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
            if detect_silence(audio_chunk):
                self.silence_duration += CHUNK_DURATION_MS
            else:
                self.silence_duration = 0  # Reset silence counter
            
            # Check if we should transcribe
            if self.silence_duration >= SILENCE_TIMEOUT_MS and len(self.audio_buffer) > 0:
                await self._transcribe_buffer(stt_processor)
                
        except Exception as e:
            print(f"❌ Error in STT processing: {e}")
    
    async def _transcribe_buffer(self, stt_processor):
        """Transcribe the current audio buffer"""
        if len(self.audio_buffer) == 0:
            return
        
        self.segment_count += 1
        
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
        
        try:
            # Convert audio buffer to numpy array
            audio_array = np.array(audio_without_silence, dtype=np.float32)
            
            # Transcribe using the STT processor
            result = await stt_processor.transcribe(audio_array.tobytes())
            transcribed_text = result.strip()
            
            if transcribed_text:
                # Update the last text change timestamp
                self.last_text_change_timestamp.value = time.time()
                
                # Send text to the queue
                try:
                    self.text_queue.put(transcribed_text, block=False)
                    print(f"📤 Sent transcribed text to LLM: '{transcribed_text}'")
                except Exception as e:
                    print(f"❌ Failed to send text to LLM process: {e}")
            
            # Reset buffer and silence counter
            self.audio_buffer = []
            self.silence_duration = 0
            
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            # Reset buffer and silence counter on error
            self.audio_buffer = []
            self.silence_duration = 0
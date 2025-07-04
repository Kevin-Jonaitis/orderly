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

from processors.stt import RealTimeSTTProcessor

# Audio configuration
SAMPLE_RATE = 16000  # Match frontend and NeMo STT sample rate
CHUNK_DURATION_MS = 80  # 80ms chunks for real-time processing
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1280 samples at 16kHz

def record_mic_buffer(mic_record_buffer, audio_data, last_record_time, sample_rate=16000, record_seconds=5, record_filename="debug_mic_input.wav"):
    """Accumulate audio data, keep last N seconds, and save to file every N seconds."""
    mic_record_buffer = np.concatenate([mic_record_buffer, audio_data])
    max_samples = sample_rate * record_seconds
    if len(mic_record_buffer) > max_samples:
        mic_record_buffer = mic_record_buffer[-max_samples:]
    now = time.time()
    if now - last_record_time > record_seconds:
        sf.write(record_filename, mic_record_buffer, sample_rate)
        print(f"[DEBUG] Saved last {record_seconds}s of mic input to {record_filename}")
        last_record_time = now
    return mic_record_buffer, last_record_time

class STTAudioProcess(multiprocessing.Process):
    """Process that handles WebRTC audio input and STT processing"""
    
    def __init__(self, text_queue, webrtc_audio_queue, last_text_change_timestamp):
        super().__init__()
        self.text_queue = text_queue
        self.webrtc_audio_queue = webrtc_audio_queue  # Queue to receive WebRTC audio
        self.last_text_change_timestamp = last_text_change_timestamp
        self.last_text = ""  # Track previous text for comparison
        
    def run(self):
        """Main process loop - processes WebRTC audio and runs STT"""
        print("🎤 Starting STT+WebRTC Audio process...")
        
        # Initialize STT processor - let it crash if it fails
        print("📝 Initializing STT processor...")
        stt_processor = RealTimeSTTProcessor()
        print("✅ STT processor loaded")
        
        print("🎤 STT+WebRTC process started")
        print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
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
        """Async loop for processing WebRTC audio chunks"""
        audio_buffer = np.zeros(0, dtype=np.float32)
        mic_record_buffer = np.zeros(0, dtype=np.float32)
        last_record_time = time.time()
        RECORD_SECONDS = 5
        RECORD_FILENAME = "debug_mic_input.wav"
        raw_record_buffer = np.zeros(0, dtype=np.float32)
        last_raw_record_time = time.time()
        RAW_RECORD_FILENAME = "debug_raw_input.wav"
        while True:
            try:
                # Get audio data from WebRTC queue with timeout
                audio_data = self.webrtc_audio_queue.get()
                if not isinstance(audio_data, np.ndarray):
                    print("DATA NOT IN THE RIGHT FORMAT?")
                    audio_data = np.array(audio_data, dtype=np.float32)

                # Try to get sample rate from aiortc frame if available
                sample_rate = SAMPLE_RATE
                if hasattr(audio_data, 'sample_rate'):
                    sample_rate = audio_data.sample_rate
                # If frame is available in scope, use its sample_rate
                if 'frame' in locals() and hasattr(frame, 'sample_rate'):
                    sample_rate = frame.sample_rate
                print(f"[DEBUG] Using sample_rate: {sample_rate}")

                # If data is int16, scale to float32
                if audio_data.dtype == np.int16:
                    print("[DEBUG] Detected int16 audio, scaling to float32 [-1, 1]")
                    audio_data = audio_data.astype(np.float32) / 32768.0

                # Debug: print shape, dtype, min, max, and first 10 samples
                print("[DEBUG] audio_data shape:", audio_data.shape, "dtype:", audio_data.dtype)
                print("[DEBUG] audio_data min:", np.min(audio_data), "max:", np.max(audio_data))
                print("[DEBUG] audio_data first 10 samples:", audio_data[:10])

                # --- Record raw input as soon as it is received ---
                raw_record_buffer = np.concatenate([raw_record_buffer, audio_data])
                max_raw_samples = sample_rate * RECORD_SECONDS
                if len(raw_record_buffer) > max_raw_samples:
                    raw_record_buffer = raw_record_buffer[-max_raw_samples:]
                now = time.time()
                if now - last_raw_record_time > RECORD_SECONDS:
                    # Save as float32
                    sf.write(RAW_RECORD_FILENAME, raw_record_buffer, sample_rate)
                    # Save as int16 for comparison
                    sf.write(RAW_RECORD_FILENAME.replace('.wav', '_int16.wav'), (raw_record_buffer * 32768).astype(np.int16), sample_rate)
                    print(f"[DEBUG] Saved last {RECORD_SECONDS}s of RAW input to {RAW_RECORD_FILENAME} at {sample_rate} Hz")
                    last_raw_record_time = now

                # Accumulate samples in buffer
                audio_buffer = np.concatenate([audio_buffer, audio_data])

                # --- Call mic recording function right before processing chunk ---
                # Only record the audio that will be processed
                if len(audio_buffer) >= CHUNK_SIZE:
                    chunk = audio_buffer[:CHUNK_SIZE]
                    mic_record_buffer, last_record_time = record_mic_buffer(
                        mic_record_buffer, chunk, last_record_time, SAMPLE_RATE, RECORD_SECONDS, RECORD_FILENAME
                    )

                # Process all full chunks in the buffer
                while len(audio_buffer) >= CHUNK_SIZE:
                    chunk = audio_buffer[:CHUNK_SIZE]
                    audio_buffer = audio_buffer[CHUNK_SIZE:]
                    await self._process_audio_chunk(chunk, stt_processor)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing WebRTC audio: {e}")
                continue
    
    async def _process_audio_chunk(self, audio_chunk, stt_processor):
        """Process a single audio chunk with STT"""
        try:
            processing_start = time.time()
            
            # Time the entire process_audio_chunk function
            result = await stt_processor.process_audio_chunk(audio_chunk)
            
            text, internal_process_time = result
            
            # Strip text once here
            text = text.strip()
            
            if text:
                # Check if text has changed from previous output
                if text != self.last_text:
                    # Update the last text change timestamp
                    self.last_text_change_timestamp.value = time.time()
                    self.last_text = text
                
                print(f"📝 STT (WebRTC): '{text}'")
                
                # Send already-stripped text to LLM process - crash if queue full
                try:
                    self.text_queue.put(text, block=False)
                except Exception as e:
                    print(f"❌ Failed to send text to LLM process: {e}")
                    
        except Exception as e:
            print(f"❌ Error in STT processing: {e}")
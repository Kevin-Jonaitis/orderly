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

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stt import RealTimeSTTProcessor

# Audio configuration
SAMPLE_RATE = 16000  # Match frontend and NeMo STT sample rate
CHUNK_DURATION_MS = 80  # 80ms chunks for real-time processing
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1280 samples at 16kHz

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
        while True:
            try:
                # Get audio data from WebRTC queue with timeout
                audio_data = self.webrtc_audio_queue.get()

                # Debug: print raw audio data and type
                print("Raw audio_data received:", audio_data)
                print("Type:", type(audio_data))

                if not isinstance(audio_data, np.ndarray):
                    print("NOT THE INSTANCE TYPE WE THOUGHT")
                    audio_data = np.array(audio_data, dtype=np.float32)
                audio_data = audio_data.astype(np.float32)

                # Check chunk size
                if len(audio_data) != CHUNK_SIZE:
                    print(f"Warning: received chunk of size {len(audio_data)}, expected {CHUNK_SIZE}")
                    continue

                # Process chunk with STT
                await self._process_audio_chunk(audio_data, stt_processor)

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
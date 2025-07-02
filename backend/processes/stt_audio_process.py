"""
STT + Audio Process
Combines audio capture and speech-to-text processing in a single process.
"""

import sys
import os
import asyncio
import multiprocessing
import numpy as np
import queue

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stt import RealTimeSTTProcessor

# Audio configuration
SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

class STTAudioProcess(multiprocessing.Process):
    """Process that handles audio capture and STT processing"""
    
    def __init__(self, text_queue):
        super().__init__()
        self.text_queue = text_queue
        
    def run(self):
        """Main process loop - runs audio capture and STT"""
        # Import sounddevice only in the child process to avoid multiprocessing conflicts
        import sounddevice as sd
        
        # Initialize STT processor - let it crash if it fails
        print("📝 Initializing STT processor...")
        stt_processor = RealTimeSTTProcessor()
        print("✅ STT processor loaded")
        
        # Set up audio capture with thread-safe queue
        audio_queue = queue.Queue(maxsize=100)
        
        def audio_callback(indata, frames, time, status):
            """Audio callback - convert to mono and queue"""
            if status:
                print(f"⚠️  Audio status: {status}")
            mono_data = indata[:, 0].astype(np.float32).copy()
            try:
                audio_queue.put_nowait(mono_data)  # Let it crash if queue full
            except queue.Full:
                print("⚠️  Audio queue full, dropping frame")
        
        print("🎤 STT+Audio process started")
        print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
        
        # Try to use the default device with explicit configuration
        try:
            print("🎤 Attempting to open audio input stream...")
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=audio_callback,
                blocksize=CHUNK_SIZE,  # 80ms blocks (1920 samples at 24kHz)
                device=None,  # Use default device
            ):
                print("✅ Audio stream opened successfully")
                # Run async event loop
                asyncio.run(self._process_audio_loop(stt_processor, audio_queue))
        except Exception as e:
            print(f"❌ Failed to open audio stream: {e}")
            print("Available devices:")
            print(sd.query_devices())
            raise
    
    async def _process_audio_loop(self, stt_processor, audio_queue):
        """Async loop for processing audio chunks"""
        while True:
            # Get audio chunk from thread-safe queue (blocking)
            audio_chunk = audio_queue.get(block=True)
            
            # Process through STT
            text = await stt_processor.process_audio_chunk(audio_chunk)
            
            if text.strip():
                # Send text to LLM process - crash if queue full
                try:
                    self.text_queue.put(text.strip(), block=False)
                except Exception as e:
                    print(f"❌ Failed to send text to LLM process: {e}")
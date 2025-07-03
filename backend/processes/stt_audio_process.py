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
import time

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.stt import RealTimeSTTProcessor

# Audio configuration
SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

class STTAudioProcess(multiprocessing.Process):
    """Process that handles audio capture and STT processing"""
    
    def __init__(self, text_queue, stt_process_time, stt_total_chunk_time, 
                 audio_capture_delay, last_word_processing_time, last_text_change_time):
        super().__init__()
        self.text_queue = text_queue
        self.stt_process_time = stt_process_time              # STT internal processing time
        self.stt_total_chunk_time = stt_total_chunk_time      # Complete function time
        self.audio_capture_delay = audio_capture_delay        # Audio capture to processing delay
        self.last_word_processing_time = last_word_processing_time  # Frame-to-text time
        self.last_text_change_time = last_text_change_time    # When STT text last changed
        self.last_text = ""  # Track previous text for comparison
        
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
            """Audio callback - convert to mono and queue with timestamp"""
            if status:
                print(f"⚠️  Audio status: {status}")
            mono_data = indata[:, 0].astype(np.float32).copy()
            capture_timestamp = time.currentTime  # Sounddevice timestamp
            try:
                audio_queue.put_nowait((mono_data, capture_timestamp))  # Include timestamp
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
            # Get audio chunk with timestamp from thread-safe queue (blocking)
            audio_chunk, capture_time = audio_queue.get(block=True)
            processing_start = time.time()
            
            # Calculate audio capture to processing delay
            audio_to_processing_delay = (processing_start - capture_time) * 1000
            self.audio_capture_delay.value = audio_to_processing_delay
            
            # Time the entire process_audio_chunk function
            chunk_start_time = time.time()
            result = await stt_processor.process_audio_chunk(audio_chunk)
            chunk_total_time = (time.time() - chunk_start_time) * 1000
            
            text, internal_process_time = result
            
            # Strip text once here
            text = text.strip()
            
            if text:
                # Check if text has changed from previous output
                if text != self.last_text:
                    # Update the last text change timestamp
                    self.last_text_change_time.value = time.time()
                    self.last_text = text
                
                # Calculate complete audio frame → text processing time
                # From original audio capture timestamp to text completion
                frame_to_text_time = (time.time() - capture_time) * 1000
                
                # Store all timing measurements
                self.stt_process_time.value = internal_process_time      # Internal model processing
                self.stt_total_chunk_time.value = chunk_total_time      # Complete function time
                self.last_word_processing_time.value = frame_to_text_time  # Frame→text time
                
                # Enhanced logging with frame-to-text timing
                print(f"📊 STT Timing - Frame→Text: {frame_to_text_time:.1f}ms, "
                      f"Capture→Process: {audio_to_processing_delay:.1f}ms, "
                      f"Internal: {internal_process_time:.1f}ms, Total: {chunk_total_time:.1f}ms")
                
                # Send already-stripped text to LLM process - crash if queue full
                try:
                    self.text_queue.put(text, block=False)
                except Exception as e:
                    print(f"❌ Failed to send text to LLM process: {e}")
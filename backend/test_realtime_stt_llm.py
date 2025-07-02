#!/usr/bin/env python3
"""
Real-time STT → LLM Pipeline
Combines streaming STT with LLM processing with KV cache preservation.
"""

import sys
import os
import time
import asyncio
import argparse
import signal
import numpy as np
import sounddevice as sd
from pathlib import Path
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import processors
from processors.stt import RealTimeSTTProcessor  
from processors.llm import LLMReasoner
import torch
import logging

# Set up minimal logging to reduce noise
logging.getLogger('nemo').setLevel(logging.WARNING)

SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

class RealTimeSTTLLMProcessor:
    """Combined real-time STT → LLM processor with KV cache management"""
    
    def __init__(self):
        print("🚀 Initializing combined STT → LLM pipeline...")
        start_init = time.time()
        
        # Initialize STT processor
        print("📝 Loading STT processor...")
        self.stt_processor = RealTimeSTTProcessor()
        print("✅ STT processor loaded successfully")
        
        # Initialize LLM processor
        print("🧠 Loading LLM processor...")
        self.llm_reasoner = LLMReasoner()
        print("✅ LLM processor loaded successfully")
        
        # Request management
        self.current_llm_task = None
        # Track current LLM task
        self.accumulated_text = ""
        self.last_unique_text = ""  # Track last unique text to detect new content
        
        init_time = (time.time() - start_init) * 1000
        print(f"✅ Combined STT → LLM pipeline initialized in {init_time:.0f}ms")
    
    async def process_audio_chunk(self, audio_chunk):
        """Process 80ms chunks - delegate to STT processor and handle LLM integration"""
        try:
            # Get transcription from STT processor
            current_text = await self.stt_processor.process_audio_chunk(audio_chunk)
            
            # Handle STT output and LLM pipeline
            if current_text.strip():
                # Only process if this is actually NEW text
                if current_text.strip() != self.last_unique_text:
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] FOUND NEW TEXT: '{current_text.strip()}' (was: '{self.last_unique_text}')")
                    # Genuinely new text detected
                    self.last_unique_text = current_text.strip()
                    # Replace accumulated text instead of appending (STT gives cumulative results)
                    self.accumulated_text = current_text.strip()
                    
                    # Cancel existing LLM and start new processing immediately (non-blocking)
                    # Cancel any existing LLM task
                    if self.current_llm_task and not self.current_llm_task.done():
                        print("🚫 Cancelling previous LLM task...")
                        self.current_llm_task.cancel()
                    else:
                        print("NO PREVIOUS LLM TASK FOUND")
                    
                    # Create new LLM task immediately (no debouncing)
                    self.current_llm_task = asyncio.create_task(self._llm_processing(None, current_text.strip()))
                else:
                    # Same text as before, don't cancel LLM
                    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] TEXT UNCHANGED: '{current_text.strip()}' == '{self.last_unique_text}'")
            
        except Exception as e:
            print(f"❌ Error processing audio chunk: {e}")
    
    async def _llm_processing(self, _, text_to_process: str):
        """Process text with LLM immediately"""
        try:
            print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🧠 Sending to LLM: '{text_to_process}'")
            
            # Generate streaming response using simplified LLMReasoner call
            response_start = time.time()
            
            full_response = ""
            for token in self.llm_reasoner.generate_response_stream(text_to_process, None):
                full_response += token
                # Print streaming output in real-time
                print(token, end='', flush=True)
            
            response_time = (time.time() - response_start) * 1000
            print(f"\n✅ LLM response ({response_time:.0f}ms): '{full_response.strip()}'")
            
            # Clear accumulated text after processing
            self.accumulated_text = ""
            
        except asyncio.CancelledError:
            print("🚫 LLM request cancelled")
            raise
        except Exception as e:
            print(f"❌ Error in LLM processing: {e}")

    async def _mock_llm_stream(self, text_to_process: str):
        """Mock LLM streaming to test if concurrency issues are llama-cpp specific"""
        mock_response = f"I understand you said '{text_to_process}'. This is a mock response to test concurrent processing."
        
        # Split into tokens (words and punctuation)
        import re
        tokens = re.findall(r'\w+|\W+', mock_response)
        
        for i, token in enumerate(tokens):
            # Small delay between tokens to simulate LLM processing
            await asyncio.sleep(0.05)  # 50ms per token
            yield token
            
            # Every few tokens, show progress
            if i % 5 == 0:
                print(f"\n[MOCK] Processing token {i+1}/{len(tokens)}", flush=True)

async def stream_microphone_realtime(device_id=None, show_devices=False):
    """Stream microphone audio with real-time STT → LLM processing"""
    
    if show_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Initialize combined STT → LLM processor
    processor = RealTimeSTTLLMProcessor()
    
    print(f"🎙️  Starting real-time STT → LLM pipeline...")
    print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
    print("🛑 Press Ctrl+C to stop")
    
    audio_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio input - matches Rust server behavior"""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Convert to mono float32 and add to queue (same as Rust server)
        mono_data = indata[:, 0].astype(np.float32).copy()
        loop.call_soon_threadsafe(audio_queue.put_nowait, mono_data)
    
    # Set input device if specified
    if device_id is not None:
        sd.default.device[0] = device_id
    
    # Start audio streaming with same parameters as Rust server
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=CHUNK_SIZE,  # 80ms blocks (1920 samples at 24kHz)
    ):
        try:
            chunk_count = 0
            while True:
                # Get 80ms audio chunk from queue
                audio_chunk = await audio_queue.get()
                chunk_count += 1
                
                # Process each 80ms chunk through STT → LLM pipeline
                await processor.process_audio_chunk(audio_chunk)
                
                # Show processing progress
                if chunk_count % 50 == 0:  # Every ~4 seconds
                    print(f"📊 Processed {chunk_count} chunks ({chunk_count * 80}ms audio)")
                
        except KeyboardInterrupt:
            print("\n🛑 Real-time STT → LLM pipeline stopped")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")

def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupted by user")
    sys.exit(0)

async def main():
    parser = argparse.ArgumentParser(description="Real-time STT → LLM pipeline (80ms chunks)")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Run real-time streaming
    await stream_microphone_realtime(
        device_id=args.device,
        show_devices=args.list_devices
    )

if __name__ == "__main__":
    asyncio.run(main())
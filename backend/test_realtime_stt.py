#!/usr/bin/env python3
"""
Real-time STT test with modular processor system
Supports both NeMo and Faster Whisper processors
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
import multiprocessing

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import STT factory
from processors.stt import create_stt_processor
import logging

# Set up minimal logging to reduce noise
logging.getLogger('nemo').setLevel(logging.WARNING)
logging.getLogger('faster_whisper').setLevel(logging.WARNING)

SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

async def keyboard_input_handler(stop_talking_timestamp, last_text_change_timestamp):
    """Handle keyboard input for timing measurements"""
    print("⌨️  Press ENTER to mark when you stop talking")
    print("⌨️  Press ENTER again to measure time since last text change")
    print("⌨️  Press Ctrl+C to exit")
    
    while True:
        try:
            # Wait for Enter key press
            await asyncio.get_event_loop().run_in_executor(None, input)
            
            current_time = time.time()
            
            if stop_talking_timestamp.value == 0:
                # First Enter press - mark when user stopped talking
                stop_talking_timestamp.value = current_time
                print(f"⏱️  MARKED: You stopped talking at {datetime.fromtimestamp(current_time).strftime('%H:%M:%S.%f')[:-3]}")
                
            else:
                # Second Enter press - measure time difference between last text change and stop talking
                if last_text_change_timestamp.value > 0 and stop_talking_timestamp.value > 0:
                    # Calculate time from last text change to when user stopped talking
                    time_diff = (last_text_change_timestamp.value - stop_talking_timestamp.value) * 1000  # Convert to ms
                    print(f"⏱️  TIME FROM LAST TEXT CHANGE TO STOP TALKING: {time_diff:.1f}ms")
                else:
                    print("⚠️  Need both text change and stop talking timestamps")
                
                # Reset the stop talking timestamp for next measurement
                stop_talking_timestamp.value = 0
                print("⌨️  Press ENTER to mark when you stop talking again")
                
        except KeyboardInterrupt:
            print("\n🛑 Keyboard input handler stopped")
            break
        except Exception as e:
            print(f"❌ Error in keyboard input: {e}")

async def stream_microphone_realtime(device_id=None, show_devices=False, processor_type="nemo", model_size="base"):
    """Stream microphone audio with real-time 80ms chunk processing"""
    
    if show_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Initialize STT processor using factory
    print(f"🚀 Initializing {processor_type.upper()} STT processor...")
    try:
        if processor_type == "faster-whisper":
            stt_processor = create_stt_processor(processor_type, model_size=model_size)
        else:
            stt_processor = create_stt_processor(processor_type)
        print(f"✅ {processor_type.upper()} STT processor loaded successfully")
    except Exception as e:
        print(f"❌ Failed to initialize {processor_type} STT processor: {e}")
        return
    
    print(f"🎙️  Starting real-time transcription with {processor_type.upper()}...")
    print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
    print("🛑 Press Ctrl+C to stop")
    
    # Shared timestamps for timing measurements
    stop_talking_timestamp = multiprocessing.Value('d', 0.0)  # When user marks they stopped talking
    last_text_change_timestamp = multiprocessing.Value('d', 0.0)  # When text last changed
    last_text = ""  # Track previous text for comparison
    
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
    
    # Start keyboard input handler as a separate task
    keyboard_task = asyncio.create_task(
        keyboard_input_handler(stop_talking_timestamp, last_text_change_timestamp)
    )
    
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
                
                # Process each 80ms chunk directly
                result = await stt_processor.process_audio_chunk(audio_chunk)
                
                # Handle result (text, processing_time)
                if isinstance(result, tuple):
                    text, processing_time = result
                else:
                    # Backward compatibility for processors that return just text
                    text = result
                    processing_time = 0.0
                
                # Track text changes for timing measurements
                if text.strip() and text.strip() != last_text:
                    last_text_change_timestamp.value = time.time()
                    last_text = text.strip()
                    print(f"📝 TEXT CHANGED: '{last_text}' at {datetime.fromtimestamp(last_text_change_timestamp.value).strftime('%H:%M:%S.%f')[:-3]}")
                
                # Show processing progress
                if chunk_count % 50 == 0:  # Every ~4 seconds
                    print(f"📊 Processed {chunk_count} chunks ({chunk_count * 80}ms audio)")
                
        except KeyboardInterrupt:
            print("\n🛑 Real-time transcription stopped")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")
        finally:
            # Cancel keyboard task
            keyboard_task.cancel()
            try:
                await keyboard_task
            except asyncio.CancelledError:
                pass

def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupted by user")
    sys.exit(0)

async def main():
    parser = argparse.ArgumentParser(description="Real-time microphone transcription with modular STT processors")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
	
    parser.add_argument(
        "--processor", type=str, default="nemo", choices=["nemo", "faster-whisper"],
        help="STT processor to use (default: nemo)"
    )
    parser.add_argument(
        "--model-size", type=str, default="base", 
        choices=["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"],
        help="Model size for Faster Whisper (default: base)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Run real-time streaming
    await stream_microphone_realtime(
        device_id=args.device,
        show_devices=args.list_devices,
        processor_type=args.processor,
        model_size=args.model_size
    )

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Minimal TTS audio test - based on working test_wsl_audio_latency.py
This tests TTS chunks with the same simple structure that produces immediate audio
"""

import pickle
import sounddevice as sd
import queue
import time as time_module
import numpy as np

def test_tts_minimal():
    """Test TTS chunks with minimal complexity (based on working latency test)"""
    print("🎵 Loading TTS debug chunks...")
    
    # Load the same debug chunks as TTS process (exact same method)
    try:
        chunks = pickle.load(open('debug_chunks_1751577529.pkl', 'rb'))
        print(f"✅ Loaded {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Failed to load chunks: {e}")
        return
    
    # Use EXACT same audio setup as working test_wsl_audio_latency.py
    audio_queue = queue.Queue()
    first_callback_time = None
    callback_count = 0
    
    def audio_callback(outdata, frames, time, status):
        """Same callback structure as working test"""
        nonlocal first_callback_time, callback_count
        
        callback_count += 1
        current_time = time_module.time()
        
        if first_callback_time is None:
            first_callback_time = current_time
            print(f"🔊 FIRST CALLBACK: {current_time:.3f}")
        
        try:
            audio_chunk = audio_queue.get_nowait()
            # Flatten same as TTS process
            audio_chunk = audio_chunk.flatten()
            outdata[:, 0] = audio_chunk
            print(f"🔊 Callback {callback_count}: consumed chunk at {current_time:.3f}")
        except queue.Empty:
            outdata.fill(0)
            print(f"🔇 Callback {callback_count}: queue empty at {current_time:.3f}")
    
    # Same stream config as working test (and TTS process)
    stream = sd.OutputStream(
        callback=audio_callback,
        samplerate=24000,
        channels=1,
        dtype='float32',
        blocksize=2048,  # Same as TTS process
        latency=0.001    # Same as TTS process
    )
    
    print(f"🔊 Stream config: blocksize={stream.blocksize}, latency={stream.latency}")
    
    # Queue chunks BEFORE starting stream (like working test)
    input("Press Enter to queue TTS chunks and start stream...")
    
    print(f"🎵 Queuing {len(chunks)} chunks...")
    start_time = time_module.time()
    
    # Queue all chunks immediately
    for i, chunk in enumerate(chunks):
        audio_queue.put(chunk)
        if i == 0:
            first_queue_time = time_module.time()
            print(f"🎵 First chunk queued at: {first_queue_time:.3f}")
    
    end_time = time_module.time()
    queue_duration = (end_time - start_time) * 1000
    print(f"🎵 All chunks queued in {queue_duration:.1f}ms")
    
    # Start stream AFTER chunks are queued
    stream.start()
    print("▶️  Stream started with chunks ready...")
    
    # Wait for playback to complete
    print("🎵 Playing audio...")
    input("Press Enter when audio finishes...")
    
    # Cleanup
    stream.stop()
    stream.close()
    print("✅ Test complete!")

if __name__ == "__main__":
    test_tts_minimal()
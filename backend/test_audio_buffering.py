#!/usr/bin/env python3
"""
Test for audio buffering latency
Compare different approaches to isolate where the 1449ms delay comes from
"""

import sounddevice as sd
import numpy as np
import time
import threading
import queue

def test_immediate_vs_buffered():
    """Test immediate audio vs buffered streaming"""
    
    # Generate test audio
    duration = 0.1
    sample_rate = 24000
    frequency = 1000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print("🔍 AUDIO BUFFERING TEST")
    print("=" * 40)
    
    # Test 1: Immediate sd.play() (known to work)
    print("\nTEST 1: Immediate sd.play() - Should be instant")
    input("Press Enter to start...")
    start_time = time.time()
    print(f"⏰ sd.play() called: {start_time:.3f}")
    sd.play(beep, samplerate=sample_rate, blocksize=64, latency='low')
    sd.wait()
    print("Did you hear it immediately?\n")
    
    # Test 2: OutputStream with single write (minimal buffering)
    print("TEST 2: OutputStream with single chunk - Testing stream buffering")
    input("Press Enter to start...")
    
    start_time = time.time()
    print(f"⏰ OutputStream starting: {start_time:.3f}")
    
    with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                        blocksize=64, latency='low') as stream:
        print(f"🔍 Stream info: blocksize={stream.blocksize}, latency={stream.latency}")
        
        write_time = time.time()
        print(f"⏰ stream.write() called: {write_time:.3f}")
        stream.write(beep)
        
        print(f"⏰ stream.write() returned: {time.time():.3f}")
        
        # Keep stream alive to let audio play
        time.sleep(1)
    
    print("Did you hear it immediately after 'stream.write() called'?\n")
    
    # Test 3: OutputStream with queue (exact TTS replication)
    print("TEST 3: OutputStream with queue - Exact TTS method")
    input("Press Enter to start...")
    
    audio_queue = queue.Queue()
    
    def audio_thread():
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                            blocksize=64, latency='low') as stream:
            queue_get_time = time.time()
            print(f"⏰ audio_queue.get(): {queue_get_time:.3f}")
            
            audio_chunk = audio_queue.get()
            
            write_time = time.time()
            print(f"⏰ stream.write() called: {write_time:.3f}")
            stream.write(audio_chunk)
            
            print(f"⏰ stream.write() returned: {time.time():.3f}")
            time.sleep(1)  # Keep alive
    
    thread = threading.Thread(target=audio_thread)
    thread.start()
    
    queue_time = time.time()
    print(f"⏰ audio_queue.put(): {queue_time:.3f}")
    audio_queue.put(beep)
    
    thread.join()
    print("Did you hear it immediately after 'stream.write() called'?\n")
    
    # Test 4: Different buffer sizes
    print("TEST 4: Different buffer sizes")
    for blocksize in [32, 64, 128, 256, 512]:
        print(f"\nTesting blocksize={blocksize}")
        input("Press Enter...")
        
        start_time = time.time()
        print(f"⏰ Audio start: {start_time:.3f}")
        
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                            blocksize=blocksize, latency='low') as stream:
            print(f"🔍 Actual blocksize: {stream.blocksize}")
            stream.write(beep)
            time.sleep(0.5)
        
        print(f"Blocksize {blocksize}: immediate?")

def test_audio_system_info():
    """Get detailed audio system information"""
    print("\n🔍 AUDIO SYSTEM DETAILED INFO")
    print("=" * 40)
    
    # Default settings
    print(f"Default device: {sd.default.device}")
    print(f"Default samplerate: {sd.default.samplerate}")
    print(f"Default blocksize: {sd.default.blocksize}")
    print(f"Default latency: {sd.default.latency}")
    
    # All devices
    print("\nAll audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"  {i}: {device['name']} (out: {device['max_output_channels']} ch, {device['default_samplerate']} Hz)")
    
    # Test creating stream with different settings
    print("\nTesting stream creation with different latency settings:")
    
    test_settings = [
        {'latency': 'low'},
        {'latency': 'high'},
        {'latency': None},
        {'latency': 0.01},  # 10ms
        {'latency': 0.05},  # 50ms
    ]
    
    for settings in test_settings:
        try:
            with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                                blocksize=64, **settings) as stream:
                print(f"  {settings}: blocksize={stream.blocksize}, latency={stream.latency}, device={stream.device}")
        except Exception as e:
            print(f"  {settings}: FAILED - {e}")

if __name__ == "__main__":
    print("Choose test:")
    print("1. Buffering tests")
    print("2. Audio system info")
    print("3. Both")
    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice in ["1", "3"]:
        test_immediate_vs_buffered()
    
    if choice in ["2", "3"]:
        test_audio_system_info()
#!/usr/bin/env python3
"""
Audio Latency Diagnostic Tool
Compare different audio configurations to isolate the 1449ms delay source
"""

import sounddevice as sd
import numpy as np
import time
import threading
import queue

def test_audio_configurations():
    """Test various audio configurations to find the latency source"""
    
    print("🔍 AUDIO LATENCY DIAGNOSTIC")
    print("="*50)
    
    # Generate test audio
    duration = 0.3
    sample_rate = 24000
    frequency = 1000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print(f"📊 Available audio devices:")
    devices = sd.query_devices()
    print(devices)
    print()
    
    # Test 1: Direct sd.play() (like the working beep test)
    print("TEST 1: Direct sd.play() - Should be immediate")
    input("Press Enter to start...")
    start_time = time.time()
    print(f"⏰ Audio started: {start_time:.3f}")
    sd.play(beep, samplerate=sample_rate, blocksize=64, latency='low')
    sd.wait()
    print("Listen for the beep and note timing\n")
    
    # Test 2: Threading + Queue (like TTS current method)
    print("TEST 2: Threading + Queue (TTS current method)")
    input("Press Enter to start...")
    
    audio_queue = queue.Queue()
    playback_started = threading.Event()
    
    def threaded_audio():
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                            blocksize=64, latency='low') as stream:
            audio_chunk = audio_queue.get()
            playback_time = time.time()
            print(f"🔊 Stream write started: {playback_time:.3f}")
            playback_started.set()
            stream.write(audio_chunk)
    
    thread = threading.Thread(target=threaded_audio)
    thread.start()
    
    queue_time = time.time()
    print(f"⏰ Queuing audio: {queue_time:.3f}")
    audio_queue.put(beep)
    
    playback_started.wait()
    thread.join()
    print("Listen for the beep and note timing\n")
    
    # Test 3: Different buffer sizes
    print("TEST 3: Different buffer sizes")
    for blocksize in [32, 64, 128, 256, 512, 1024]:
        print(f"Testing blocksize={blocksize}")
        input("Press Enter...")
        start_time = time.time()
        print(f"⏰ Audio started: {start_time:.3f}")
        sd.play(beep, samplerate=sample_rate, blocksize=blocksize, latency='low')
        sd.wait()
        print()
    
    # Test 4: Different sample rates
    print("TEST 4: Different sample rates")
    for rate in [24000, 44100, 48000]:
        test_beep = beep
        if rate != 24000:
            # Resample
            test_beep = np.interp(np.linspace(0, len(beep), int(len(beep) * rate/24000)), 
                                np.arange(len(beep)), beep).astype(np.float32)
        
        print(f"Testing sample rate={rate}Hz")
        input("Press Enter...")
        start_time = time.time()
        print(f"⏰ Audio started: {start_time:.3f}")
        sd.play(test_beep, samplerate=rate, blocksize=64, latency='low')
        sd.wait()
        print()
    
    # Test 5: Different latency settings
    print("TEST 5: Different latency settings")
    for latency_setting in ['low', 'high', None]:
        print(f"Testing latency='{latency_setting}'")
        input("Press Enter...")
        start_time = time.time()
        print(f"⏰ Audio started: {start_time:.3f}")
        if latency_setting:
            sd.play(beep, samplerate=sample_rate, blocksize=64, latency=latency_setting)
        else:
            sd.play(beep, samplerate=sample_rate, blocksize=64)
        sd.wait()
        print()

def test_tts_exact_replication():
    """Replicate the exact TTS audio method to isolate the issue"""
    
    print("🎯 TTS METHOD REPLICATION TEST")
    print("="*40)
    
    # Generate test audio chunks (simulate TTS streaming)
    duration = 0.1  # Small chunks like TTS
    sample_rate = 24000
    frequency = 1000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    chunk = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print("Replicating exact TTS audio streaming method...")
    input("Press Enter to start...")
    
    # Exact replication of TTS process
    audio_queue = queue.Queue()
    
    def audio_callback():
        """Exact copy of TTS audio_callback"""
        with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                            blocksize=64, latency='low') as stream:
            first_playback = True
            while True:
                audio_chunk = audio_queue.get()
                if audio_chunk is None:
                    break
                
                if first_playback:
                    playback_time = time.time()
                    print(f"🔊 ACTUAL PLAYBACK STARTED: {playback_time:.3f}")
                    first_playback = False
                
                stream.write(audio_chunk.squeeze())
    
    # Start audio thread
    audio_thread = threading.Thread(target=audio_callback)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Queue several chunks (like TTS does)
    queue_start = time.time()
    print(f"⏰ First chunk queued: {queue_start:.3f}")
    
    for i in range(5):  # Multiple chunks like TTS
        audio_queue.put(chunk)
        if i == 0:
            print(f"📥 Chunk {i+1} queued: {time.time():.3f}")
        time.sleep(0.01)  # Small delay between chunks
    
    time.sleep(2)  # Let it play
    audio_queue.put(None)  # Stop signal
    print("Listen for the beep and note timing")

if __name__ == "__main__":
    print("Choose test:")
    print("1. Full configuration tests")
    print("2. TTS exact replication")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        test_audio_configurations()
    elif choice == "2":
        test_tts_exact_replication()
    else:
        print("Invalid choice")
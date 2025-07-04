#!/usr/bin/env python3
"""
PyAudio Low-Latency Audio Test
Direct PortAudio access for minimal latency audio playback testing.
Should achieve <100ms latency compared to ~200ms with sounddevice.
"""

import time
import pickle
import threading
import pyaudio
import numpy as np

# Global variables for timing
hear_audio_timestamp = None

def wait_for_audio_input():
    """Wait for user to press Enter when they hear audio"""
    global hear_audio_timestamp
    input("🔊 Press Enter when you HEAR the first audio...")
    hear_audio_timestamp = time.time()
    print(f"🔊 Audio heard at: {hear_audio_timestamp:.3f}")

def load_debug_chunks():
    """Load debug chunks for testing"""
    try:
        chunks = pickle.load(open('debug_beep_chunks_1751591457.pkl', 'rb'))
        print(f"🎵 [PyAudio] Loading {len(chunks)} debug chunks...")
        return chunks
    except Exception as e:
        print(f"❌ [PyAudio] Failed to load debug chunks: {e}")
        return None

def test_pyaudio_latency():
    """Test PyAudio for minimal latency audio playback"""
    global hear_audio_timestamp
    
    print("🎵 Testing PyAudio Low-Latency Audio")
    print("🎧 Audio timing test ready")
    
    # Start input listener thread immediately
    input_thread = threading.Thread(target=wait_for_audio_input)
    input_thread.daemon = True
    input_thread.start()
    
    # Load debug chunks
    print("🔧 Loading debug chunks...")
    chunks = load_debug_chunks()
    if not chunks:
        print("❌ Failed to load debug chunks - aborting test")
        return
    
    # Initialize PyAudio with PulseAudio backend for WSL2
    print("🔧 Initializing PyAudio...")
    import os
    os.environ['ALSA_PCM_CARD'] = 'default'
    os.environ['ALSA_PCM_DEVICE'] = '0'
    p = pyaudio.PyAudio()
    
    # Print available audio devices for debugging
    print("🔊 Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"   [{i}] {info['name']} - {info['maxOutputChannels']}ch")
    
    try:
        # Create audio stream with minimal latency settings
        print("🔧 Creating PyAudio stream...")
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=512,  # Much smaller than sounddevice's 2048
            output_device_index=None,  # Use default device
            start=False  # Don't start immediately
        )
        
        print(f"🔊 PyAudio stream configuration:")
        print(f"   Sample rate: 24000 Hz")
        print(f"   Buffer size: 512 samples ({512/24000*1000:.1f}ms)")
        print(f"   Format: Float32")
        print(f"   Channels: 1")
        
        # Start the stream
        stream.start_stream()
        print("🔧 PyAudio stream started...")
        
        # Record timing right before sending audio data
        debug_chunks_sent_timestamp = time.time()
        print(f"⏰ Debug chunks sent at: {debug_chunks_sent_timestamp:.3f}")
        
        # Write chunks directly to stream (blocking mode)
        print("🎵 Writing audio chunks...")
        start_write = time.time()
        
        for i, chunk in enumerate(chunks):
            # Convert to bytes for PyAudio
            audio_bytes = chunk.astype(np.float32).tobytes()
            stream.write(audio_bytes)
            
            if i == 0:
                first_write_time = time.time()
                print(f"🔊 First chunk written at: {first_write_time:.3f}")
        
        end_write = time.time()
        write_duration = (end_write - start_write) * 1000
        print(f"🎵 All chunks written in {write_duration:.1f}ms")
        
        print("⏳ Chunks written - waiting for audio input...")
        
        # Wait for user input and calculate timing
        input_thread.join()
        
        if hear_audio_timestamp:
            total_latency_ms = (hear_audio_timestamp - debug_chunks_sent_timestamp) * 1000
            first_write_latency_ms = (hear_audio_timestamp - first_write_time) * 1000
            
            print(f"\n🎯 PYAUDIO TIMING RESULTS:")
            print(f"   Debug chunks sent: {debug_chunks_sent_timestamp:.3f}")
            print(f"   First chunk written: {first_write_time:.3f}")
            print(f"   Audio heard: {hear_audio_timestamp:.3f}")
            print(f"   Total latency (chunks sent → heard): {total_latency_ms:.1f}ms")
            print(f"   Audio latency (first write → heard): {first_write_latency_ms:.1f}ms")
            print(f"   Buffer size: 512 samples ({512/24000*1000:.1f}ms)")
            print(f"   Expected improvement: ~100-150ms vs sounddevice")
        else:
            print("❌ No audio timing recorded")
        
        # Keep stream alive briefly for cleanup
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ PyAudio error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up PyAudio resources
        try:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
            print("✅ PyAudio cleanup complete")
        except:
            pass

def compare_with_sounddevice():
    """Print comparison information"""
    print("\n📊 PyAudio vs sounddevice Comparison:")
    print("=" * 50)
    print("PyAudio advantages:")
    print("  • Direct PortAudio access (no wrapper)")
    print("  • Smaller buffer size (512 vs 2048 samples)")
    print("  • Lower-level control")
    print("  • Expected latency: ~50-100ms")
    print("\nsounddevice comparison:")
    print("  • Higher-level Python wrapper")
    print("  • Larger default buffer size")
    print("  • Measured latency: ~200ms")

if __name__ == "__main__":
    print("🧪 PyAudio Low-Latency Audio Test")
    print("=" * 50)
    
    compare_with_sounddevice()
    print()
    
    test_pyaudio_latency()
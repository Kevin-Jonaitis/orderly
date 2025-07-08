#!/usr/bin/env python3
"""
Standalone test for AudioProcessor - tests debug chunk playback
This should achieve ~200ms latency like test_tts_audio_minimal.py
"""

import multiprocessing
import time
import pickle
import threading

# CRITICAL: Set multiprocessing start method to 'spawn' for compatibility
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

from processes.tts_audio_process import AudioProcessor

# Global variables for timing
hear_audio_timestamp = None

def wait_for_audio_input():
    """Wait for user to press Enter when they hear audio"""
    global hear_audio_timestamp
    input("🔊 Press Enter when you HEAR the first audio...")
    hear_audio_timestamp = time.time()
    print(f"🔊 Audio heard at: {hear_audio_timestamp:.3f}")

def load_and_queue_debug_chunks(audio_queue):
    """Load and queue debug chunks for testing (moved from AudioProcessor)"""
    try:
        chunks = pickle.load(open('debug_beep_chunks_1751591457.pkl', 'rb'))
        print(f"🎵 [TEST] Loading {len(chunks)} debug chunks...")
        
        start_time = time.time()
        
        # Queue all chunks immediately (like TTSProcess does)
        for i, chunk in enumerate(chunks):
            audio_queue.put(chunk)
            if i == 0:
                first_queue_time = time.time()
                print(f"🎵 [TEST] First chunk queued at: {first_queue_time:.3f}")
        
        end_time = time.time()
        queue_duration = (end_time - start_time) * 1000
        print(f"🎵 [TEST] All chunks queued in {queue_duration:.1f}ms")
        
        return True
    except Exception as e:
        print(f"❌ [TEST] Failed to load debug chunks: {e}")
        return False


def test_audio_processor_blocking():
    """Test AudioProcessor with blocking mode"""
    global hear_audio_timestamp
    
    print("🎵 Testing AudioProcessor with blocking mode...")
    print("🎧 Audio timing test ready")
    
    # Start input listener thread immediately
    input_thread = threading.Thread(target=wait_for_audio_input)
    input_thread.daemon = True
    input_thread.start()
    
    # Create multiprocessing queue and timestamp
    audio_queue = multiprocessing.Queue(maxsize=100)
    first_audio_chunk_timestamp = multiprocessing.Value('d', 0.0)
    
    # Create AudioProcessor (no debug mode needed)
    audio_processor = AudioProcessor(
        audio_queue=audio_queue,
        first_audio_chunk_timestamp=first_audio_chunk_timestamp,
        use_blocking_audio=True  # Test blocking mode
    )
    
    print("🔧 Starting AudioProcessor in blocking mode...")
    
    try:
        # Start the audio processor
        audio_processor.start()
        
        print("⏳ AudioProcessor started - now loading chunks...")
        
        # Record timing right before loading and queuing chunks
        debug_chunks_sent_timestamp = time.time()
        print(f"⏰ Debug chunks sent at: {debug_chunks_sent_timestamp:.3f}")
        
        # Load and queue debug chunks AFTER starting AudioProcessor
        print("🔧 Loading debug chunks...")
        if not load_and_queue_debug_chunks(audio_queue):
            print("❌ Failed to load debug chunks - aborting test")
            return
        
        print("⏳ Chunks loaded - waiting for audio input...")
        
        # Wait for user input and calculate timing
        input_thread.join()
        
        if hear_audio_timestamp:
            latency_ms = (hear_audio_timestamp - debug_chunks_sent_timestamp) * 1000
            print(f"\n🎯 TIMING RESULTS:")
            print(f"   Debug chunks sent: {debug_chunks_sent_timestamp:.3f}")
            print(f"   Audio heard: {hear_audio_timestamp:.3f}")
            print(f"   Total latency: {latency_ms:.1f}ms")
        else:
            print("❌ No audio timing recorded")
        
        # Keep process alive a bit longer for cleanup
        time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping AudioProcessor test...")
    finally:
        audio_processor.terminate()
        audio_processor.join(timeout=2)
        
        if audio_processor.is_alive():
            print("🔪 Force killing AudioProcessor")
            audio_processor.kill()
        
        print("✅ AudioProcessor blocking test complete")

if __name__ == "__main__":
    print("🧪 AudioProcessor Standalone Tests")
    print("=" * 50)
    
    test_audio_processor_blocking()
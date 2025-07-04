#!/usr/bin/env python3
"""
Standalone test for AudioProcessor - tests debug chunk playback
This should achieve ~200ms latency like test_tts_audio_minimal.py
"""

import multiprocessing
import time

# CRITICAL: Set multiprocessing start method to 'spawn' for compatibility
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

from processes.audio_process import AudioProcessor

def test_audio_processor_debug():
    """Test AudioProcessor with debug chunks (standalone)"""
    print("🎵 Testing AudioProcessor with debug chunks...")
    
    # Create multiprocessing queue and timestamp
    audio_queue = multiprocessing.Queue(maxsize=100)
    first_audio_chunk_timestamp = multiprocessing.Value('d', 0.0)
    
    # Create AudioProcessor in debug mode
    audio_processor = AudioProcessor(
        audio_queue=audio_queue,
        first_audio_chunk_timestamp=first_audio_chunk_timestamp,
        use_blocking_audio=False,  # Test callback mode first
        debug_mode=True  # Enable debug mode for standalone testing
    )
    
    print("🔧 Starting AudioProcessor in debug mode...")
    print("   This should load debug chunks and play them with ~200ms latency")
    print("   (similar to test_tts_audio_minimal.py)")
    
    try:
        # Start the audio processor
        audio_processor.start()
        
        print("⏳ AudioProcessor started - you should hear audio soon...")
        print("   Press Ctrl+C to stop")
        
        # Keep main process alive
        audio_processor.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping AudioProcessor test...")
        audio_processor.terminate()
        audio_processor.join(timeout=2)
        
        if audio_processor.is_alive():
            print("🔪 Force killing AudioProcessor")
            audio_processor.kill()
        
        print("✅ AudioProcessor test complete")

def test_audio_processor_blocking():
    """Test AudioProcessor with blocking mode"""
    print("🎵 Testing AudioProcessor with blocking mode...")
    
    # Create multiprocessing queue and timestamp
    audio_queue = multiprocessing.Queue(maxsize=100)
    first_audio_chunk_timestamp = multiprocessing.Value('d', 0.0)
    
    # Create AudioProcessor in blocking mode
    audio_processor = AudioProcessor(
        audio_queue=audio_queue,
        first_audio_chunk_timestamp=first_audio_chunk_timestamp,
        use_blocking_audio=True,  # Test blocking mode
        debug_mode=True  # Enable debug mode for standalone testing
    )
    
    print("🔧 Starting AudioProcessor in blocking mode...")
    
    try:
        # Start the audio processor
        audio_processor.start()
        
        print("⏳ AudioProcessor started - you should hear audio soon...")
        print("   Press Ctrl+C to stop")
        
        # Keep main process alive
        audio_processor.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping AudioProcessor test...")
        audio_processor.terminate()
        audio_processor.join(timeout=2)
        
        if audio_processor.is_alive():
            print("🔪 Force killing AudioProcessor")
            audio_processor.kill()
        
        print("✅ AudioProcessor blocking test complete")

if __name__ == "__main__":
    print("🧪 AudioProcessor Standalone Tests")
    print("=" * 50)
    
    print("\nChoose test mode:")
    print("1. Callback mode (default)")
    print("2. Blocking mode")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        test_audio_processor_blocking()
    else:
        test_audio_processor_debug()
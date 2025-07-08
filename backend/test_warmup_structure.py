#!/usr/bin/env python3
"""
Test script to verify STT warm-up structure with shared flag
"""

import sys
import os
import multiprocessing

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_warmup_structure():
    """Test that the warm-up structure with shared flag is correct"""
    print("🧪 Testing STT warm-up structure with shared flag...")
    
    # Test that the STT audio process can be created with the warm-up flag
    try:
        from processes.stt_audio_process import STTAudioProcess
        print("✅ STTAudioProcess imported successfully")
        
        # Create a dummy warm-up flag
        stt_warmup_flag = multiprocessing.Value('i', 0)
        
        # Test that we can create the process with the flag
        # We'll use dummy queues for testing
        dummy_queue = multiprocessing.Queue()
        dummy_timestamp = multiprocessing.Value('d', 0.0)
        
        stt_process = STTAudioProcess(
            text_queue=dummy_queue,
            webrtc_audio_queue=dummy_queue,
            last_text_change_timestamp=dummy_timestamp,
            manual_speech_end_timestamp=dummy_timestamp,
            stt_warmup_flag=stt_warmup_flag
        )
        print("✅ STTAudioProcess created successfully with warm-up flag")
        
    except Exception as e:
        print(f"❌ Error testing STTAudioProcess: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ STT warm-up structure test completed successfully")
    return True

if __name__ == "__main__":
    success = test_warmup_structure()
    if success:
        print("🎉 All structure tests passed!")
    else:
        print("❌ Some structure tests failed!")
        sys.exit(1) 
#!/usr/bin/env python3
"""
Multi-Process STT → LLM → TTS Pipeline
Runs STT+Audio, LLM processing, and TTS in separate processes with fail-fast design.
"""

import sys
import os
import time
import multiprocessing

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# This must be done before importing any CUDA-related modules
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import process classes
from processes.stt_audio_process import STTAudioProcess
from processes.llm_process import LLMProcess
from processes.tts_process import TTSProcess

def main():
    print("🚀 Starting Multi-Process STT → LLM → TTS Pipeline")
    
    # Create communication queues
    text_queue = multiprocessing.Queue(maxsize=100)  # STT → LLM
    tts_text_queue = multiprocessing.Queue(maxsize=10)  # LLM → TTS
    
    # Start processes
    stt_process = STTAudioProcess(text_queue)
    llm_process = LLMProcess(text_queue, tts_text_queue)  # Now takes both queues
    tts_process = TTSProcess(tts_text_queue)
    
    try:
        stt_process.start()
        llm_process.start()
        tts_process.start()
        
        print("🎙️ STT → LLM → TTS Pipeline running. Press Ctrl+C to stop.")
        
        # Wait for Ctrl+C - let everything else crash
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Send shutdown signal to TTS process
        tts_text_queue.put(None)
        
        # Kill processes immediately
        stt_process.terminate()
        llm_process.terminate()
        tts_process.terminate()
        
        stt_process.join(timeout=2)
        llm_process.join(timeout=2)
        tts_process.join(timeout=2)
        
        if stt_process.is_alive():
            print("🔪 Force killing STT process")
            stt_process.kill()
        if llm_process.is_alive():
            print("🔪 Force killing LLM process")
            llm_process.kill()
        if tts_process.is_alive():
            print("🔪 Force killing TTS process")
            tts_process.kill()
        
        print("✅ All processes terminated")

if __name__ == "__main__":
    main()
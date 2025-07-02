#!/usr/bin/env python3
"""
Multi-Process STT → LLM Pipeline
Runs STT+Audio and LLM processing in separate processes with fail-fast design.
"""

import sys
import os
import time
import multiprocessing

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import process classes
from processes.stt_audio_process import STTAudioProcess
from processes.llm_process import LLMProcess

def main():
    print("🚀 Starting Multi-Process STT → LLM Pipeline")
    
    # Create simple text queue
    text_queue = multiprocessing.Queue(maxsize=100)
    
    # Start processes
    stt_process = STTAudioProcess(text_queue)
    llm_process = LLMProcess(text_queue)
    
    try:
        stt_process.start()
        llm_process.start()
        
        print("🎙️ STT → LLM Pipeline running. Press Ctrl+C to stop.")
        
        # Wait for Ctrl+C - let everything else crash
        while True:
            time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Kill processes immediately
        stt_process.terminate()
        llm_process.terminate()
        
        stt_process.join(timeout=2)
        llm_process.join(timeout=2)
        
        if stt_process.is_alive():
            print("🔪 Force killing STT process")
            stt_process.kill()
        if llm_process.is_alive():
            print("🔪 Force killing LLM process")
            llm_process.kill()
        
        print("✅ All processes terminated")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Multi-Process STT → LLM → TTS Pipeline
Runs STT+Audio, LLM processing, and TTS in separate processes with fail-fast design.
"""

import sys
import os
import time
import multiprocessing
import threading

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

def keyboard_listener(manual_speech_end_time, manual_audio_heard_time, manual_event_count):
    """Simple keyboard listener using input() in a thread"""
    print("⌨️  Manual timing ready - In a separate terminal, you can run:")
    print("   python -c \"import time; input('Press Enter when you STOP talking: '); print(f'Speech end: {time.time()}')\"")
    print("   python -c \"import time; input('Press Enter when you HEAR audio: '); print(f'Audio heard: {time.time()}')\"")
    print()
    print("⌨️  Or use this simple interface:")
    print("   Type 's' + Enter when you stop talking")
    print("   Type 'h' + Enter when you hear the FIRST SOUND (not when speech ends)")
    print("   Type 'r' + Enter to reset")
    
    try:
        while True:
            try:
                user_input = input().strip().lower()
                current_time = time.time()
                
                if user_input == 's':
                    manual_speech_end_time.value = current_time
                    manual_event_count.value = 1
                    print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                    print("🚨🚨🚨 SPEECH END RECORDED!!! 🚨🚨🚨")
                    print(f"🚨🚨🚨 TIMESTAMP: {current_time:.3f} 🚨🚨🚨")
                    print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                    
                elif user_input == 'h':
                    if manual_speech_end_time.value > 0:
                        manual_audio_heard_time.value = current_time
                        manual_event_count.value = 2
                        print("🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
                        print("🔊🔊🔊 AUDIO HEARD RECORDED!!! 🔊🔊🔊")
                        print(f"🔊🔊🔊 TIMESTAMP: {current_time:.3f} 🔊🔊🔊")
                        print(f"🔊🔊🔊 STORED IN SHARED VAR: {manual_audio_heard_time.value:.3f} 🔊🔊🔊")
                        print("🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
                        
                        # Calculate and display manual timing
                        manual_latency = (current_time - manual_speech_end_time.value) * 1000
                        print(f"⌨️  [MANUAL] Total perceived latency: {manual_latency:.1f}ms")
                        print(f"⌨️  [MANUAL] Speech end was: {manual_speech_end_time.value:.3f}")
                        print("⌨️  Latency breakdown will be shown after TTS completes...")
                    else:
                        print("❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
                        print("❌❌❌ PLEASE PRESS 's' FIRST! ❌❌❌")
                        print("❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
                        
                elif user_input == 'r':
                    manual_event_count.value = 0
                    manual_speech_end_time.value = 0.0
                    manual_audio_heard_time.value = 0.0
                    print("🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄")
                    print("🔄🔄🔄 RESET COMPLETE!!! 🔄🔄🔄")
                    print("🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄🔄")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("⌨️  Manual timing stopped")
        pass

def main():
    print("🚀 Starting Multi-Process STT → LLM → TTS Pipeline")
    
    # Create communication queues
    text_queue = multiprocessing.Queue(maxsize=100)  # STT → LLM
    tts_text_queue = multiprocessing.Queue(maxsize=10)  # LLM → TTS
    
    # Create shared timestamps for latency tracking
    stt_process_time = multiprocessing.Value('d', 0.0)         # STT internal processing time in ms
    stt_total_chunk_time = multiprocessing.Value('d', 0.0)     # STT complete function time in ms
    audio_capture_delay = multiprocessing.Value('d', 0.0)      # Audio capture to processing delay in ms
    last_word_processing_time = multiprocessing.Value('d', 0.0) # Last word frame-to-text time in ms
    last_text_change_time = multiprocessing.Value('d', 0.0)    # When STT text last changed
    llm_queue_wait_time = multiprocessing.Value('d', 0.0)      # LLM queue waiting time in ms
    llm_to_tts_time = multiprocessing.Value('d', 0.0)          # LLM time-to-TTS in ms
    llm_total_time = multiprocessing.Value('d', 0.0)           # Total LLM processing time in ms
    timestamp_llm_start = multiprocessing.Value('d', 0.0)
    timestamp_llm_complete = multiprocessing.Value('d', 0.0)
    timestamp_tts_start = multiprocessing.Value('d', 0.0)
    timestamp_first_audio = multiprocessing.Value('d', 0.0)
    
    # Manual timing variables for spacebar events
    manual_speech_end_time = multiprocessing.Value('d', 0.0)   # When user stops talking
    manual_audio_heard_time = multiprocessing.Value('d', 0.0)  # When user hears audio
    manual_event_count = multiprocessing.Value('i', 0)         # Event counter (0,1,2)
    
    # Start processes
    stt_process = STTAudioProcess(text_queue, stt_process_time, stt_total_chunk_time, 
                                  audio_capture_delay, last_word_processing_time, last_text_change_time)
    llm_process = LLMProcess(text_queue, tts_text_queue, timestamp_llm_start, timestamp_llm_complete, 
                            llm_to_tts_time, llm_total_time, llm_queue_wait_time)
    tts_process = TTSProcess(tts_text_queue, stt_process_time, stt_total_chunk_time, audio_capture_delay,
                            last_word_processing_time, last_text_change_time, llm_queue_wait_time, 
                            llm_to_tts_time, llm_total_time, timestamp_llm_start, timestamp_llm_complete, 
                            timestamp_tts_start, timestamp_first_audio, manual_speech_end_time, manual_audio_heard_time)
    
    # Start keyboard listener thread for manual timing
    keyboard_thread = threading.Thread(target=keyboard_listener, 
                                       args=(manual_speech_end_time, manual_audio_heard_time, manual_event_count))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
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
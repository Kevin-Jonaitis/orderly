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
from processes.audio_process import AudioProcessor

def keyboard_listener(manual_speech_end_timestamp, manual_audio_heard_timestamp, manual_event_count):
    """Simple keyboard listener using input() in a thread"""
    print("⌨️  Manual timing ready - In a separate terminal, you can run:")
    print("   python -c \"import time; input('Press Enter when you STOP talking: '); print(f'Speech end: {time.time()}')\"")
    print("   python -c \"import time; input('Press Enter when you HEAR audio: '); print(f'Audio heard: {time.time()}')\"")
    print()
    print("⌨️  Or use this simple interface:")
    print("   Type 's' + Enter when you stop talking")
    print("   Type 'h' + Enter when you hear the FIRST SOUND (not when speech ends)")
    
    try:
        while True:
            try:
                user_input = input().strip().lower()
                current_time = time.time()
                
                if user_input == 's':
                    manual_speech_end_timestamp.value = current_time
                    manual_event_count.value = 1
                    print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                    print("🚨🚨🚨 SPEECH END RECORDED!!! 🚨🚨🚨")
                    print(f"🚨🚨🚨 TIMESTAMP: {current_time:.3f} 🚨🚨🚨")
                    print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨")
                    
                elif user_input == 'h':
                    if manual_speech_end_timestamp.value > 0:
                        manual_audio_heard_timestamp.value = current_time
                        manual_event_count.value = 2
                        print("🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
                        print("🔊🔊🔊 AUDIO HEARD RECORDED!!! 🔊🔊🔊")
                        print(f"🔊🔊🔊 TIMESTAMP: {current_time:.3f} 🔊🔊🔊")
                        print(f"🔊🔊🔊 STORED IN SHARED VAR: {manual_audio_heard_timestamp.value:.3f} 🔊🔊🔊")
                        print("🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
                        
                        # Calculate and display manual timing
                        manual_latency = (current_time - manual_speech_end_timestamp.value) * 1000
                        print(f"⌨️  [MANUAL] Total perceived latency: {manual_latency:.1f}ms")
                        print(f"⌨️  [MANUAL] Speech end was: {manual_speech_end_timestamp.value:.3f}")
                        print("⌨️  Latency breakdown will be shown after TTS completes...")
                    else:
                        print("❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
                        print("❌❌❌ PLEASE PRESS 's' FIRST! ❌❌❌")
                        print("❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌❌")
                        
                    
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
    audio_queue = multiprocessing.Queue(maxsize=100)  # TTS → AudioProcessor
    
    # Create 7 simple timing variables
    manual_speech_end_timestamp = multiprocessing.Value('d', 0.0)
    manual_audio_heard_timestamp = multiprocessing.Value('d', 0.0)
    last_text_change_timestamp = multiprocessing.Value('d', 0.0)
    llm_start_timestamp = multiprocessing.Value('d', 0.0)
    llm_send_to_tts_timestamp = multiprocessing.Value('d', 0.0)
    llm_complete_timestamp = multiprocessing.Value('d', 0.0)
    first_audio_chunk_timestamp = multiprocessing.Value('d', 0.0)
    
    # Manual timing event counter
    manual_event_count = multiprocessing.Value('i', 0)         # Event counter (0,1,2)
    
    # Start processes
    stt_process = STTAudioProcess(text_queue, last_text_change_timestamp)
    llm_process = LLMProcess(text_queue, tts_text_queue, llm_start_timestamp, llm_send_to_tts_timestamp, llm_complete_timestamp)
    tts_process = TTSProcess(tts_text_queue, audio_queue, first_audio_chunk_timestamp)
    audio_process = AudioProcessor(audio_queue, first_audio_chunk_timestamp, use_blocking_audio=True, debug_mode=False)
    
    # Start keyboard listener thread for manual timing
    keyboard_thread = threading.Thread(target=keyboard_listener, 
                                       args=(manual_speech_end_timestamp, manual_audio_heard_timestamp, manual_event_count))
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    try:
        stt_process.start()
        llm_process.start()
        tts_process.start()
        audio_process.start()
        
        print("🎙️ STT → LLM → TTS → Audio Pipeline running. Press Ctrl+C to stop.")
        
        # Timing display - wait for manual_audio_heard_timestamp to be set, then print once
        last_audio_heard_value = 0.0
        while True:
            # Only print when manual_audio_heard_timestamp changes from 0
            if manual_audio_heard_timestamp.value > 0 and manual_audio_heard_timestamp.value != last_audio_heard_value:
                last_audio_heard_value = manual_audio_heard_timestamp.value
                
                # Display timing breakdown
                print("\n🚨🚨🚨 TIMING BREAKDOWN 🚨🚨🚨")
                print(f"📊 manual_speech_end_timestamp: {manual_speech_end_timestamp.value:.3f}")
                print(f"📊 last_text_change_timestamp: {last_text_change_timestamp.value:.3f}")
                print(f"📊 llm_start_timestamp: {llm_start_timestamp.value:.3f}")
                print(f"📊 llm_send_to_tts_timestamp: {llm_send_to_tts_timestamp.value:.3f}")
                print(f"📊 llm_complete_timestamp: {llm_complete_timestamp.value:.3f}")
                print(f"📊 first_audio_chunk_timestamp: {first_audio_chunk_timestamp.value:.3f}")
                print(f"📊 manual_audio_heard_timestamp: {manual_audio_heard_timestamp.value:.3f}")
                
                # Calculate intervals
                total_latency = (manual_audio_heard_timestamp.value - manual_speech_end_timestamp.value) * 1000
                print(f"⚡ TOTAL LATENCY: {total_latency:.1f}ms")
                
                speech_to_text = (last_text_change_timestamp.value - manual_speech_end_timestamp.value) * 1000
                print(f"🎤 Speech End → Text Finished Generating: {speech_to_text:.1f}ms")
                
                text_to_llm = (llm_start_timestamp.value - last_text_change_timestamp.value) * 1000
                print(f"🧠 Text → LLM Start: {text_to_llm:.1f}ms")
                
                llm_processing = (llm_send_to_tts_timestamp.value - llm_start_timestamp.value) * 1000
                print(f"🧠 LLM Processing: {llm_processing:.1f}ms")
                
                llm_to_first_audio = (first_audio_chunk_timestamp.value - llm_send_to_tts_timestamp.value) * 1000
                print(f"🎵 LLM Send to TTS -> First Audio chunk: {llm_to_first_audio:.1f}ms")
                
                audio_to_heard = (manual_audio_heard_timestamp.value - first_audio_chunk_timestamp.value) * 1000
                print(f"🔊 First Audio Chunk → Heard Audio: {audio_to_heard:.1f}ms")
                
                print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨\n")
                
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        
        # Send shutdown signal to TTS process
        tts_text_queue.put(None)
        
        # Kill processes immediately
        stt_process.terminate()
        llm_process.terminate()
        tts_process.terminate()
        audio_process.terminate()
        
        stt_process.join(timeout=2)
        llm_process.join(timeout=2)
        tts_process.join(timeout=2)
        audio_process.join(timeout=2)
        
        if stt_process.is_alive():
            print("🔪 Force killing STT process")
            stt_process.kill()
        if llm_process.is_alive():
            print("🔪 Force killing LLM process")
            llm_process.kill()
        if tts_process.is_alive():
            print("🔪 Force killing TTS process")
            tts_process.kill()
        if audio_process.is_alive():
            print("🔪 Force killing Audio process")
            audio_process.kill()
        
        print("✅ All processes terminated")

if __name__ == "__main__":
    main()
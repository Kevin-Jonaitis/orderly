#!/usr/bin/env python3

import time
import threading
import queue
from multiprocessing import Process
import sounddevice as sd
from processors.orpheus_tts import OrpheusTTS


class TTSProcess(Process):
    """TTS process that receives text from LLM and streams audio to speakers.
    
    Modeled after test_tts.py with real-time audio streaming.
    Runs in a separate process to avoid blocking the LLM.
    """
    
    def __init__(self, tts_text_queue, stt_process_time, stt_total_chunk_time, audio_capture_delay,
                 last_word_processing_time, last_text_change_time, llm_queue_wait_time, llm_to_tts_time, llm_total_time, 
                 timestamp_llm_start, timestamp_llm_complete, timestamp_tts_start, timestamp_first_audio,
                 manual_speech_end_time, manual_audio_heard_time):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.tts = None
        # All timing data for latency tracking
        self.stt_process_time = stt_process_time            # STT internal processing time in ms
        self.stt_total_chunk_time = stt_total_chunk_time    # STT complete function time in ms
        self.audio_capture_delay = audio_capture_delay      # Audio capture to processing delay in ms
        self.last_word_processing_time = last_word_processing_time  # Frame-to-text time in ms
        self.last_text_change_time = last_text_change_time  # When STT text last changed
        self.llm_queue_wait_time = llm_queue_wait_time      # LLM queue waiting time in ms
        self.llm_to_tts_time = llm_to_tts_time              # LLM time-to-TTS in ms
        self.llm_total_time = llm_total_time                # Total LLM time in ms
        self.timestamp_llm_start = timestamp_llm_start
        self.timestamp_llm_complete = timestamp_llm_complete
        self.timestamp_tts_start = timestamp_tts_start
        self.timestamp_first_audio = timestamp_first_audio
        # Manual timing data
        self.manual_speech_end_time = manual_speech_end_time    # When user stops talking
        self.manual_audio_heard_time = manual_audio_heard_time  # When user hears audio
    
    def run(self):
        """Main TTS process loop - initialize TTS and process text from queue"""
        print("🎵 Starting TTS process...")
        
        # Initialize TTS system with integrated models (like test_tts.py)
        print("🔧 Initializing OrpheusTTS in TTS process...")
        self.tts = OrpheusTTS()
        print("✅ TTS process initialization complete")
        
        # Set up audio streaming (identical to test_tts.py --stream)
        print("🎵 Setting up real-time audio streaming...")
        audio_queue = queue.Queue()
        
        def audio_callback():
            """Stream audio chunks to speakers"""
            with sd.OutputStream(samplerate=24000, channels=1, dtype='float32') as stream:
                while True:
                    audio_chunk = audio_queue.get()  # Block until chunk available
                    if audio_chunk is None:  # End signal
                        break
                    stream.write(audio_chunk.squeeze())
        
        # Start audio streaming thread
        audio_thread = threading.Thread(target=audio_callback)
        audio_thread.daemon = True  # Die when main process dies
        audio_thread.start()
        print("🎵 Audio streaming thread started")
        
        # Main loop: wait for text from LLM and generate TTS
        print("🎧 TTS process ready - waiting for text from LLM...")
        while True:
            # Block until we receive text from LLM process
            text = self.tts_text_queue.get()
            
            # Check for shutdown signal
            if text is None:
                print("🛑 TTS process received shutdown signal")
                break
            
            # Record timestamp when TTS receives text from LLM
            self.timestamp_tts_start.value = time.time()
            
            print(f"🎤 TTS received text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate TTS audio (text is already stripped from STT/LLM)
            first_chunk = True
            for result in self.tts.tts_streaming(text, "tara", save_file=None):
                sample_rate, audio_array, chunk_count = result
                
                # Record timestamp for first audio chunk
                if first_chunk:
                    self.timestamp_first_audio.value = time.time()
                    first_chunk = False
                    # Only display pipeline latency breakdown if manual timing is complete
                    if self.manual_audio_heard_time.value > 0:
                        self._display_pipeline_latency()
                    else:
                        print("🎵 First audio chunk generated - waiting for manual timing...")
                
                # Stream to speakers
                audio_queue.put(audio_array)
            
            # Signal end of this TTS generation
            print("✅ TTS generation complete")
            
            # Check if manual timing became available during TTS generation
            if self.manual_audio_heard_time.value > 0 and self.timestamp_first_audio.value > 0:
                self._display_pipeline_latency()
        
        # Clean up audio streaming
        print("🔧 Cleaning up TTS audio streaming...")
        audio_queue.put(None)  # End signal
        time.sleep(0.5)  # Let the audio finish playing
        audio_thread.join()  # Wait for audio to finish
        print("🛑 TTS process shutdown complete")
    
    def _display_pipeline_latency(self):
        """Display comprehensive pipeline latency breakdown with manual timing comparison"""
        # Get all timing data
        last_word_frame_to_text = self.last_word_processing_time.value  # Complete frame→text time
        audio_capture_delay = self.audio_capture_delay.value            # Audio capture to processing delay
        stt_internal_time = self.stt_process_time.value                 # STT internal processing time
        stt_total_time = self.stt_total_chunk_time.value                # STT complete function time
        last_text_change = self.last_text_change_time.value             # When STT text last changed
        llm_queue_wait = self.llm_queue_wait_time.value                 # LLM queue waiting time
        llm_to_tts_time = self.llm_to_tts_time.value                    # LLM time-to-TTS
        llm_total_time = self.llm_total_time.value                      # Total LLM processing time
        tts_start = self.timestamp_tts_start.value
        first_audio = self.timestamp_first_audio.value
        
        # Get manual timing data
        manual_speech_end = self.manual_speech_end_time.value
        manual_audio_heard = self.manual_audio_heard_time.value
        
        # Calculate TTS time-to-first-audio
        tts_time_to_first_audio = (first_audio - tts_start) * 1000
        
        # Calculate new gap measurements
        speech_end_to_stt_complete = 0
        audio_generated_to_heard = 0
        
        if manual_speech_end > 0 and last_text_change > 0:
            speech_end_to_stt_complete = (last_text_change - manual_speech_end) * 1000
        
        if first_audio > 0 and manual_audio_heard > 0:
            audio_generated_to_heard = (manual_audio_heard - first_audio) * 1000
        
        # Calculate comprehensive pipeline times using frame-to-text measurement
        measured_total_frame_based = (last_word_frame_to_text + llm_queue_wait + 
                                     llm_to_tts_time + tts_time_to_first_audio)
        measured_total_component_based = (audio_capture_delay + stt_total_time + llm_queue_wait + 
                                         llm_to_tts_time + tts_time_to_first_audio)
        total_with_complete_llm = (last_word_frame_to_text + llm_queue_wait + 
                                  llm_total_time + tts_time_to_first_audio)
        
        # Calculate manual latency if both events recorded
        manual_latency = 0
        if manual_speech_end > 0 and manual_audio_heard > 0:
            manual_latency = (manual_audio_heard - manual_speech_end) * 1000
        
        # Display results VERY CLEARLY AND LOUDLY
        print("\n🚨🚨🚨 COMPREHENSIVE LATENCY BREAKDOWN 🚨🚨🚨")
        print(f"📊 Last Word Frame→Text: {last_word_frame_to_text:.1f}ms")
        print(f"📊 Audio Capture→Processing: {audio_capture_delay:.1f}ms")
        print(f"📊 STT Internal Processing: {stt_internal_time:.1f}ms")
        print(f"📊 STT Total Function Time: {stt_total_time:.1f}ms")
        print(f"📊 LLM Queue Wait Time: {llm_queue_wait:.1f}ms")
        print(f"🧠 LLM Time-to-TTS: {llm_to_tts_time:.1f}ms")
        print(f"🧠 LLM Total Processing: {llm_total_time:.1f}ms")
        print(f"🎵 TTS Time-to-First-Audio: {tts_time_to_first_audio:.1f}ms")
        print(f"⚡ MEASURED TOTAL (Frame→Audio): {measured_total_frame_based:.1f}ms")
        print(f"📈 MEASURED TOTAL (Components): {measured_total_component_based:.1f}ms")
        print(f"📈 TOTAL WITH COMPLETE LLM: {total_with_complete_llm:.1f}ms")
        
        # New gap measurements
        if speech_end_to_stt_complete > 0:
            print(f"🔍 Speech End → STT Complete: {speech_end_to_stt_complete:.1f}ms")
        
        if audio_generated_to_heard > 0:
            print(f"🔍 Audio Generated → Audio Heard: {audio_generated_to_heard:.1f}ms")
        
        # Manual timing comparison
        if manual_latency > 0:
            print(f"⌨️  MANUAL TOTAL (Speech→Audio): {manual_latency:.1f}ms")
            print(f"🔍 MANUAL vs MEASURED GAP: {manual_latency - measured_total_frame_based:.1f}ms")
            
            # Gap analysis
            if manual_latency > measured_total_frame_based:
                gap = manual_latency - measured_total_frame_based
                print(f"❓ UNACCOUNTED TIME: {gap:.1f}ms")
            else:
                print("✅ Manual timing matches measured components!")
        else:
            print("⌨️  MANUAL TIMING: Not recorded yet")
            print("   Press SPACEBAR when you stop talking, then when you hear audio")
        
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨\n")
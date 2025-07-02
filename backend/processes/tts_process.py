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
    
    def __init__(self, tts_text_queue, stt_process_time, timestamp_llm_start, 
                 timestamp_llm_complete, timestamp_tts_start, timestamp_first_audio):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.tts = None
        # All timestamps for latency tracking
        self.stt_process_time = stt_process_time  # Processing time in ms
        self.timestamp_llm_start = timestamp_llm_start
        self.timestamp_llm_complete = timestamp_llm_complete
        self.timestamp_tts_start = timestamp_tts_start
        self.timestamp_first_audio = timestamp_first_audio
    
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
            
            # Generate TTS audio (identical to test_tts.py logic)
            first_chunk = True
            for result in self.tts.tts_streaming(text.strip(), "tara", save_file=None):
                sample_rate, audio_array, chunk_count = result
                
                # Record timestamp for first audio chunk
                if first_chunk:
                    self.timestamp_first_audio.value = time.time()
                    first_chunk = False
                    # Display pipeline latency breakdown after first audio
                    self._display_pipeline_latency()
                
                # Stream to speakers
                audio_queue.put(audio_array)
            
            # Signal end of this TTS generation
            print("✅ TTS generation complete")
        
        # Clean up audio streaming
        print("🔧 Cleaning up TTS audio streaming...")
        audio_queue.put(None)  # End signal
        time.sleep(0.5)  # Let the audio finish playing
        audio_thread.join()  # Wait for audio to finish
        print("🛑 TTS process shutdown complete")
    
    def _display_pipeline_latency(self):
        """Display pipeline latency breakdown VERY CLEARLY AND LOUDLY"""
        # Get all timestamps and processing time
        stt_processing = self.stt_process_time.value  # Already in ms
        llm_start = self.timestamp_llm_start.value
        llm_complete = self.timestamp_llm_complete.value
        tts_start = self.timestamp_tts_start.value
        first_audio = self.timestamp_first_audio.value
        
        # Calculate latencies in milliseconds
        llm_processing = (llm_complete - llm_start) * 1000
        tts_time_to_first_audio = (first_audio - tts_start) * 1000
        total_pipeline_time = stt_processing + llm_processing + tts_time_to_first_audio
        
        # Display results VERY CLEARLY AND LOUDLY
        print("\n🚨🚨🚨 PIPELINE LATENCY BREAKDOWN 🚨🚨🚨")
        print(f"📊 STT Processing: {stt_processing:.1f}ms")
        print(f"🧠 LLM Processing: {llm_processing:.1f}ms")
        print(f"🎵 TTS Time-to-First-Audio: {tts_time_to_first_audio:.1f}ms")
        print(f"⚡ TOTAL PIPELINE TIME: {total_pipeline_time:.1f}ms")
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨\n")
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
    
    def __init__(self, tts_text_queue):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.tts = None
    
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
            
            print(f"🎤 TTS received text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate TTS audio (identical to test_tts.py logic)
            for result in self.tts.tts_streaming(text.strip(), "tara", save_file=None):
                sample_rate, audio_array, chunk_count = result
                
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
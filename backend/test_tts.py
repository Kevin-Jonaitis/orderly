#!/usr/bin/env python3

import argparse
import soundfile
import numpy as np
import time
import sounddevice as sd
from processors.orpheus_tts import OrpheusTTS

# OrpheusTTS class is now imported from processors

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with OrpheusTTS")
    parser.add_argument("--text", type=str, required=True, help="The text to convert to speech")
    parser.add_argument("--voice", type=str, choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"], default="tara", help="The voice to use for the TTS")
    parser.add_argument("--output", type=str, default=None, help="Output file to save audio")
    parser.add_argument("--stream", action="store_true", help="Stream audio to speakers in real-time")
    # Removed --api-url argument since we're using direct model inference
    args = parser.parse_args()

    # Initialize TTS system with integrated models
    tts = OrpheusTTS()
    
    # Set up audio streaming if requested
    if args.stream:
        print("🎵 Starting real-time audio streaming...")
        import threading
        import queue
        
        # Audio streaming setup
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
        audio_thread.start()
    
    # Generate and process audio chunks
    for result in tts.tts_streaming(args.text.strip(), args.voice, save_file=args.output):
        sample_rate, audio_array, chunk_count = result
        
        # Stream to speakers if requested
        if args.stream:
            audio_queue.put(audio_array)

    
    # Clean up streaming
    if args.stream:
        audio_queue.put(None)  # End signal
        time.sleep(0.5) # Let the audio finish playing
        audio_thread.join()    # Wait for audio to finish
        print("✅ Audio streaming complete")

if __name__ == "__main__":
    main()
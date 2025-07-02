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
    parser.add_argument("--output", type=str, default="output.wav", help="Output file")
    parser.add_argument("--save-chunks", action="store_true", help="Save individual audio chunks (for testing)")
    parser.add_argument("--chunk-prefix", type=str, default="chunk", help="Prefix for chunk filenames")
    parser.add_argument("--play", action="store_true", help="Play audio through speakers in addition to saving")
    # Removed --api-url argument since we're using direct model inference
    args = parser.parse_args()

    # Initialize TTS system with integrated models
    tts = OrpheusTTS()
    
    # Streaming mode (always enabled now)
    final_audio_data = None
    final_stats = None
    
    for result in tts.tts_streaming(args.text.strip(), args.voice, args.save_chunks, args.chunk_prefix):
        if len(result) == 3:
            # Individual chunk: (sample_rate, audio_array, chunk_count)
            sample_rate, audio_array, chunk_count = result
            # In a real streaming scenario, you would play this chunk immediately
        elif len(result) >= 4 and result[-1] is True:
            # Final combined result: (sample_rate, audio_data, total_time, audio_duration, is_final)
            sample_rate, final_audio_data, generation_time, audio_duration = result[:4]
            final_stats = (generation_time, audio_duration)
            break
    
    if final_audio_data is not None:
        soundfile.write(args.output, final_audio_data.squeeze(), sample_rate)
        generation_time, audio_duration = final_stats
        
        # Performance metrics
        real_time_factor = generation_time / audio_duration if audio_duration > 0 else float('inf')
        is_real_time = generation_time < audio_duration
        
        print(f"\n📁 Final combined audio saved to {args.output}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Real-time factor: {real_time_factor:.2f}x")
        print(f"Real-time capable: {'✅ YES' if is_real_time else '❌ NO'}")
        
        if is_real_time:
            speedup = audio_duration / generation_time
            print(f"Speed advantage: {speedup:.1f}x faster than real-time")
        
        # Play audio if requested
        if args.play:
            print("\n🔊 Playing audio...")
            try:
                # Normalize audio to float32 in range [-1, 1] for sounddevice
                audio_float = final_audio_data.astype(np.float32)
                if audio_float.max() > 1.0 or audio_float.min() < -1.0:
                    # If not already normalized, normalize it
                    audio_float = audio_float / np.max(np.abs(audio_float))
                
                sd.play(audio_float.squeeze(), sample_rate)
                sd.wait()  # Wait for playback to complete
                print("✅ Audio playback complete")
            except Exception as e:
                print(f"❌ Audio playback error: {e}")
    else:
        print("❌ No final audio generated!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import json
import requests
import soundfile
import numpy as np
import onnxruntime
import time
from typing import Generator
from decoder import convert_to_audio, turn_token_into_id, tokens_decoder_sync
import threading
import queue as queue_module
import asyncio

class OrpheusCpp:
    def __init__(self):
        # Initialize with the new torch-based decoder
        print("Initializing Torch-based SNAC decoder...")
        # The decoder module handles all model loading and warmup

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        return turn_token_into_id(token_text, index)


    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        # Use the new torch-based decoder
        return convert_to_audio(multiframe, len(multiframe))

    async def _generate_tokens_async(self, text: str, voice: str = "tara"):
        """Async generator that fetches tokens from the server."""
        text_formatted = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
        completion_url = "http://localhost:8080/v1/completions"
        data = {
            "stream": True,
            "prompt": text_formatted,
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
        }
        
        print(f"Starting token generation for: '{text_formatted}'")
        request_start = time.time()
        response = requests.post(completion_url, json=data, stream=True)
        first_token_time = None
        token_count = 0
        last_line_time = request_start
        
        try:
            for line in response.iter_lines():
                line_received_time = time.time()
                line = line.decode("utf-8")
                time_between_lines = line_received_time - last_line_time

                if line.startswith("data: ") and not line.endswith("[DONE]"):
                    try:
                        line_data = json.loads(line[len("data: "):])
                        if "choices" in line_data and len(line_data["choices"]) > 0:
                            token = line_data["choices"][0].get("text", "")
                            if token:
                                token_time = time.time()
                                if first_token_time is None:
                                    first_token_time = token_time
                                    time_to_first_token = first_token_time - request_start
                                    print(f"Time to first token: {time_to_first_token:.3f}s")
                                
                                token_count += 1
                                if token_count <= 5 or token_count % 10 == 0:
                                    print(f"Token {token_count}: received in {time_between_lines:.3f}s since last token")
                                
                                # Yield token with timing info
                                yield (token, token_time)
                    except json.JSONDecodeError:
                        continue
                
                last_line_time = line_received_time
            
            total_token_time = time.time() - request_start
            print(f"Token generation complete: {token_count} tokens in {total_token_time:.3f}s ({token_count/total_token_time:.1f} tokens/sec)")
        
        finally:
            print("GOT ALL TOKENS!")
            response.close()

    def tts(self, text: str, voice: str = "tara"):
        overall_start = time.time()
        
        print(f"\n=== TTS Generation Started ===")
        print(f"Input text: '{text}' (voice: {voice})")
        
        buffer = []
        decode_start = time.time()
        
        # Use tokens_decoder_sync with proper token queueing
        for audio_bytes in tokens_decoder_sync(self._generate_tokens_async(text, voice)):
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
            buffer.append(audio_array)
        
        decode_time = time.time() - decode_start
        total_time = time.time() - overall_start
        
        if buffer:
            audio_data = np.concatenate(buffer, axis=1)
            audio_duration = audio_data.shape[1] / 24000  # samples / sample_rate = seconds
            
            print(f"\n=== Timing Breakdown ===")
            print(f"Total processing time: {total_time:.3f}s")
            print(f"Audio decoding time: {decode_time:.3f}s")
            
            return (24000, audio_data, total_time, audio_duration)
        else:
            print("❌ No audio generated!")
            return (24000, np.array([], dtype=np.int16).reshape(1, 0), total_time, 0.0)

    def tts_streaming(self, text: str, voice: str = "tara", save_chunks: bool = False, chunk_prefix: str = "chunk"):
        overall_start = time.time()
        
        print(f"\n=== Streaming TTS Generation Started ===")
        print(f"Input text: '{text}' (voice: {voice})")
        print(f"Save individual chunks: {save_chunks}")
        
        buffer = []
        decode_start = time.time()
        chunk_count = 0
        
        for audio_bytes in tokens_decoder_sync(self._generate_tokens_async(text, voice)):
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
            buffer.append(audio_array)
            chunk_count += 1
            
            if save_chunks:
                chunk_filename = f"{chunk_prefix}_{chunk_count:03d}.wav"
                soundfile.write(chunk_filename, audio_array.squeeze(), 24000)
                print(f"💾 Saved chunk {chunk_count}: {chunk_filename}")
            
            # Yield each chunk for real-time streaming
            yield (24000, audio_array, chunk_count)
        
        decode_time = time.time() - decode_start
        total_time = time.time() - overall_start
        
        if buffer:
            # Combine all chunks into final audio
            audio_data = np.concatenate(buffer, axis=1)
            audio_duration = audio_data.shape[1] / 24000
            
            print(f"\n=== Streaming Timing Breakdown ===")
            print(f"Total chunks generated: {chunk_count}")
            print(f"Total processing time: {total_time:.3f}s")
            print(f"Audio decoding time: {decode_time:.3f}s")
            print(f"Audio duration: {audio_duration:.2f}s")
            
            yield (24000, audio_data, total_time, audio_duration, True)  # Final combined result
        else:
            print("❌ No audio generated!")
            yield (24000, np.array([], dtype=np.int16).reshape(1, 0), total_time, 0.0, True)

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with OrpheusCpp")
    parser.add_argument("--text", type=str, required=True, help="The text to convert to speech")
    parser.add_argument("--voice", type=str, choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"], default="tara", help="The voice to use for the TTS")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode")
    parser.add_argument("--save-chunks", action="store_true", help="Save individual audio chunks (for testing)")
    parser.add_argument("--chunk-prefix", type=str, default="chunk", help="Prefix for chunk filenames")
    args = parser.parse_args()

    orpheus = OrpheusCpp()
    
    if args.streaming:
        # Streaming mode
        final_audio_data = None
        final_stats = None
        
        for result in orpheus.tts_streaming(args.text.strip(), args.voice, args.save_chunks, args.chunk_prefix):
            if len(result) == 3:
                # Individual chunk: (sample_rate, audio_array, chunk_count)
                sample_rate, audio_array, chunk_count = result
                print(f"🎵 Streaming chunk {chunk_count} ready ({audio_array.shape[1]} samples)")
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
        else:
            print("❌ No final audio generated!")
    else:
        # Non-streaming mode (original behavior)
        sample_rate, samples, generation_time, audio_duration = orpheus.tts(args.text.strip(), args.voice)
        soundfile.write(args.output, samples.squeeze(), sample_rate)
        
        # Performance metrics
        real_time_factor = generation_time / audio_duration if audio_duration > 0 else float('inf')
        is_real_time = generation_time < audio_duration
        
        print(f"Audio saved to {args.output}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Generation time: {generation_time:.2f}s")
        print(f"Real-time factor: {real_time_factor:.2f}x")
        print(f"Real-time capable: {'✅ YES' if is_real_time else '❌ NO'}")
        
        if is_real_time:
            speedup = audio_duration / generation_time
            print(f"Speed advantage: {speedup:.1f}x faster than real-time")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

import argparse
import json
import requests
import soundfile
import numpy as np
import time
from typing import Generator
from decoder import convert_to_audio, turn_token_into_id, tokens_decoder_sync
import threading
import queue as queue_module
import asyncio

class OrpheusTTS:
    def __init__(self, api_url="http://localhost:8080/v1/completions"):
        """
        Initialize the OrpheusTTS system with the torch-based SNAC decoder.
        
        Args:
            api_url: The URL of the LLM inference server
        """
        self.api_url = api_url
        print("Initializing Torch-based SNAC decoder...")
        # The decoder module handles all model loading
        self.warmup_time = 0.0  # No warmup tracking in simplified version

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        """Convert token text to ID using the decoder's function."""
        return turn_token_into_id(token_text, index)

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        """Convert token frames to audio using the decoder's function."""
        # Note: convert_to_audio expects count parameter for tracking position, not length
        return convert_to_audio(multiframe, len(multiframe))

    def _generate_tokens_sync(self, text: str, voice: str = "tara"):
        """Synchronous token generator that fetches tokens from the server."""
        text_formatted = f"<|audio|>{voice}: {text}<|eot_id|>"
        
        data = {
            "stream": True,
            "prompt": text_formatted,
            "max_tokens": 2000,
    
        }
        
        print(f"Starting token generation for: '{text_formatted}'")
        request_start = time.time()
        
        try:
            response = requests.post(self.api_url, json=data, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error connecting to server at {self.api_url}: {e}")
            return
        
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
                                
                                # Yield just the token (decoder expects string, not tuple)
                                yield token
                    except json.JSONDecodeError:
                        continue
                
                last_line_time = line_received_time
            
            total_token_time = time.time() - request_start
            print(f"Token generation complete: {token_count} tokens in {total_token_time:.3f}s ({token_count/total_token_time:.1f} tokens/sec)")
        
        finally:
            print("GOT ALL TOKENS!")
            response.close()

    def tts_streaming(self, text: str, voice: str = "tara", save_chunks: bool = False, chunk_prefix: str = "chunk"):
        """
        Generate TTS audio with streaming support.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use (default: "tara")
            save_chunks: Whether to save individual chunks to disk
            chunk_prefix: Prefix for chunk filenames
            
        Yields:
            Individual chunks: (sample_rate, audio_array, chunk_count)
            Final result: (sample_rate, audio_data, total_time, audio_duration, is_final)
        """
        overall_start = time.time()
        
        print(f"\n=== Streaming TTS Generation Started ===")
        print(f"Input text: '{text}' (voice: {voice})")
        print(f"Save individual chunks: {save_chunks}")
        
        buffer = []
        first_audio_chunk_time = None
        chunk_count = 0
        last_chunk_time = overall_start  # Track time of last chunk (start with overall start)
        
        # Track token generation start
        token_gen_start = time.time()
        first_token_time = None
        
        # Create a wrapper to track token timing
        def token_gen_with_timing():
            nonlocal first_token_time
            for i, token in enumerate(self._generate_tokens_sync(text, voice)):
                if i == 0 and first_token_time is None:
                    first_token_time = time.time() - token_gen_start
                yield token
        
        decode_start = time.time()
        
        for audio_bytes in tokens_decoder_sync(token_gen_with_timing()):
            current_chunk_time = time.time()
            chunk_count += 1
            
            # Calculate time since last chunk (or start for first chunk)
            time_since_last = current_chunk_time - last_chunk_time
            
            if chunk_count == 1 and first_audio_chunk_time is None:
                first_audio_chunk_time = current_chunk_time - overall_start
                print(f"Audio chunk {chunk_count}: {time_since_last:.3f}s since start")
            else:
                print(f"Audio chunk {chunk_count}: {time_since_last:.3f}s since last chunk")
            
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).reshape(1, -1)
            buffer.append(audio_array)
            
            # Update last chunk time for next iteration
            last_chunk_time = current_chunk_time
            
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
            print(f"Time to first token: {first_token_time:.3f}s" if first_token_time else "Time to first token: N/A")
            print(f"Time to first audio chunk: {first_audio_chunk_time:.3f}s" if first_audio_chunk_time else "Time to first audio chunk: N/A")
            print(f"Audio decoding time: {decode_time:.3f}s")
            print(f"Audio duration: {audio_duration:.2f}s")
            
            yield (24000, audio_data, total_time, audio_duration, True)  # Final combined result
        else:
            print("❌ No audio generated!")
            yield (24000, np.array([], dtype=np.int16).reshape(1, 0), total_time, 0.0, True)

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with OrpheusTTS")
    parser.add_argument("--text", type=str, required=True, help="The text to convert to speech")
    parser.add_argument("--voice", type=str, choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"], default="tara", help="The voice to use for the TTS")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file")
    parser.add_argument("--save-chunks", action="store_true", help="Save individual audio chunks (for testing)")
    parser.add_argument("--chunk-prefix", type=str, default="chunk", help="Prefix for chunk filenames")
    parser.add_argument("--api-url", type=str, default="http://localhost:1234/v1/completions", help="LLM inference server URL")
    args = parser.parse_args()

    # Initialize TTS system
    tts = OrpheusTTS(api_url=args.api_url)
    
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
    else:
        print("❌ No final audio generated!")

if __name__ == "__main__":
    main()
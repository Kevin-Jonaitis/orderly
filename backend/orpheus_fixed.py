#!/usr/bin/env python3

import argparse
import json
import requests
import soundfile
import numpy as np
import onnxruntime
import time
from typing import Generator

class OrpheusCpp:
    def __init__(self):
        snac_model_path = "snac_decoder_model.onnx"
        
        # Configure session options to minimize CPU fallback and memcpy nodes
        session_options = onnxruntime.SessionOptions()
        session_options.enable_mem_reuse = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_pattern = True
        
        # Configure CUDA provider options to minimize CPU fallback
        cuda_provider_options = {
            'device_id': 0,
            'arena_extend_strategy': 'kSameAsRequested',  # Reduce memory fragmentation
            'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # Increase GPU memory limit to 4GB
            'cudnn_conv_algo_search': 'HEURISTIC',  # Faster than EXHAUSTIVE
        }
        
        providers = [
            ('CUDAExecutionProvider', cuda_provider_options),
            'CPUExecutionProvider'
        ]
        
        # Load SNAC model with optimizations
        self._snac_session = onnxruntime.InferenceSession(
            snac_model_path,
            sess_options=session_options,
            providers=providers,
        )
        print(f"ONNX providers: {self._snac_session.get_providers()}")
        
        # Warmup the model to eliminate cold start penalty
        self._warmup_model()
    
    def _warmup_model(self):
        """Warmup SNAC model with dummy inference to eliminate cold start penalty."""
        print("Warming up SNAC model...")
        warmup_start = time.time()
        
        # Create dummy inputs matching expected shape and data type (int64)
        dummy_codes_0 = np.array([[1, 2, 3, 4]], dtype=np.int64)  # 4 tokens
        dummy_codes_1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)  # 8 tokens
        dummy_codes_2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], dtype=np.int64)  # 16 tokens
        
        # Get input names and create input dict
        snac_input_names = [x.name for x in self._snac_session.get_inputs()]
        input_dict = dict(zip(snac_input_names, [dummy_codes_0, dummy_codes_1, dummy_codes_2]))
        
        try:
            # Run warmup inference
            _ = self._snac_session.run(None, input_dict)
            warmup_time = time.time() - warmup_start
            print(f"Model warmup completed in {warmup_time:.3f}s")
        except Exception as e:
            print(f"Warmup failed: {e} - proceeding anyway")

    def _token_to_id(self, token_text: str, index: int) -> int | None:
        token_string = token_text.strip()

        # Find the last token in the string
        last_token_start = token_string.rfind("<custom_token_")

        if last_token_start == -1:
            return None

        # Extract the last token
        last_token = token_string[last_token_start:]

        # Process the last token
        if last_token.startswith("<custom_token_") and last_token.endswith(">"):
            try:
                number_str = last_token[14:-1]
                token_id = int(number_str) - 10 - ((index % 7) * 4096)
                return token_id
            except ValueError:
                return None
        else:
            return None

    def _decode(
        self, token_gen: Generator[tuple[str, float, float], None, None], overall_start_time: float = None
    ) -> Generator[np.ndarray, None, None]:
        """Token decoder that converts token stream to audio stream."""
        buffer = []
        count = 0
        audio_chunks_generated = 0
        chunk_start_time = None
        tokens_for_chunk_received_time = None
        first_wav_start_time = None
        first_wav_generated = False
        
        print("Starting audio decoding...")
        
        for token_text, token_time, request_start_time in token_gen:
            # Set the first_wav_start_time to the token request start time on first iteration
            if first_wav_start_time is None:
                first_wav_start_time = request_start_time
            token = self._token_to_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    # Record when we have enough tokens for this chunk
                    tokens_for_chunk_received_time = token_time
                    buffer_to_proc = buffer[-28:]
                    
                    # Start timing the audio processing
                    audio_start = time.time()
                    time_since_tokens_ready = audio_start - tokens_for_chunk_received_time
                    
                    audio_samples = self._convert_to_audio(buffer_to_proc)
                    audio_end = time.time()
                    audio_conversion_time = audio_end - audio_start
                    
                    if audio_samples is not None:
                        audio_chunks_generated += 1
                        total_chunk_time = audio_end - tokens_for_chunk_received_time
                        wav_creation_time = audio_end - audio_start
                        
                        # Log first wav file timing
                        if not first_wav_generated:
                            time_to_first_wav = audio_end - first_wav_start_time
                            time_from_tts_start = audio_end - overall_start_time if overall_start_time else None
                            token_accumulation_time = tokens_for_chunk_received_time - first_wav_start_time
                            snac_inference_time = audio_conversion_time
                            
                            print(f"🎵 FIRST WAV FILE GENERATED!")
                            print(f"  📊 BREAKDOWN:")
                            print(f"    Token accumulation time (to get 28+ tokens): {token_accumulation_time:.3f}s")
                            print(f"    SNAC inference time: {snac_inference_time:.3f}s")
                            print(f"    Pipeline overhead: {time_since_tokens_ready:.6f}s")
                            print(f"  📈 TOTALS:")
                            print(f"    Time to first wav file creation: {time_to_first_wav:.3f}s")
                            if time_from_tts_start:
                                print(f"    Time from TTS start to first wav: {time_from_tts_start:.3f}s")
                            first_wav_generated = True
                        
                        print(f"Audio chunk {audio_chunks_generated}:")
                        print(f"  Time from tokens ready to processing start: {time_since_tokens_ready:.6f}s")
                        print(f"  Audio conversion time (SNAC inference): {audio_conversion_time:.3f}s")
                        print(f"  Wav file creation time: {wav_creation_time:.3f}s")
                        print(f"  Total chunk processing time: {total_chunk_time:.3f}s")
                        yield audio_samples
        
        print(f"Audio decoding complete: {audio_chunks_generated} audio chunks generated")

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:  # Ensure we have enough tokens
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        # Initialize empty numpy arrays with int64 dtype to match model expectations
        codes_0 = np.array([], dtype=np.int64)
        codes_1 = np.array([], dtype=np.int64)
        codes_2 = np.array([], dtype=np.int64)

        for j in range(num_frames):
            i = 7 * j
            # Append values to numpy arrays
            codes_0 = np.append(codes_0, frame[i])

            codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])

            codes_2 = np.append(
                codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]
            )

        # Reshape arrays to match the expected input format (add batch dimension)
        codes_0 = np.expand_dims(codes_0, axis=0)
        codes_1 = np.expand_dims(codes_1, axis=0)
        codes_2 = np.expand_dims(codes_2, axis=0)

        # Check that all tokens are between 0 and 4096
        if (
            np.any(codes_0 < 0)
            or np.any(codes_0 > 4096)
            or np.any(codes_1 < 0)
            or np.any(codes_1 > 4096)
            or np.any(codes_2 < 0)
            or np.any(codes_2 > 4096)
        ):
            return None

        # Create input dictionary for ONNX session
        snac_input_names = [x.name for x in self._snac_session.get_inputs()]
        input_dict = dict(zip(snac_input_names, [codes_0, codes_1, codes_2]))

        # Run inference
        audio_hat = self._snac_session.run(None, input_dict)[0]

        # Process output - EXACT pastebin slicing
        audio_np = audio_hat[:, :, 2048:4096]
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

    def _token_gen(self, text: str, voice: str = "tara") -> Generator[tuple[str, float, float], None, None]:
        text = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
        completion_url = "http://localhost:8080/v1/completions"  # Our llama-server endpoint
        data = {
            "stream": True,
            "prompt": text,
            "max_tokens": 500,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
        }
        
        print(f"Starting token generation for: '{text}'")
        request_start = time.time()
        response = requests.post(completion_url, json=data, stream=True)
        first_token_time = None
        token_count = 0
        last_line_time = request_start
        
        for line in response.iter_lines():
            line_received_time = time.time()
            line = line.decode("utf-8")
            time_between_lines = line_received_time - last_line_time

            if line.startswith("data: ") and not line.endswith("[DONE]"):
                try:
                    data = json.loads(line[len("data: "):])
                    if "choices" in data and len(data["choices"]) > 0:
                        token = data["choices"][0].get("text", "")
                        if token:
                            token_time = time.time()
                            if first_token_time is None:
                                first_token_time = token_time
                                time_to_first_token = first_token_time - request_start
                                print(f"Time to first token: {time_to_first_token:.3f}s")
                            
                            token_count += 1
                            if token_count <= 5 or token_count % 10 == 0:
                                print(f"Token {token_count}: received in {time_between_lines:.3f}s since last token")
                            yield token, token_time, request_start
                except json.JSONDecodeError:
                    continue
            
            last_line_time = line_received_time
        
        total_token_time = time.time() - request_start
        print(f"Token generation complete: {token_count} tokens in {total_token_time:.3f}s ({token_count/total_token_time:.1f} tokens/sec)")

    def tts(self, text: str, voice: str = "tara"):
        overall_start = time.time()
        
        print(f"\n=== TTS Generation Started ===")
        print(f"Input text: '{text}' (voice: {voice})")
        
        buffer = []
        decode_start = time.time()
        
        for audio_bytes in self._decode(self._token_gen(text, voice), overall_start):
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
        
        for audio_bytes in self._decode(self._token_gen(text, voice), overall_start):
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
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
        
        # Load SNAC model with optimizations
        self._snac_session = onnxruntime.InferenceSession(
            snac_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print(f"ONNX providers: {self._snac_session.get_providers()}")

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
        self, token_gen: Generator[tuple[str, float], None, None]
    ) -> Generator[np.ndarray, None, None]:
        """Token decoder that converts token stream to audio stream."""
        buffer = []
        count = 0
        audio_chunks_generated = 0
        
        print("Starting audio decoding...")
        
        for token_text, token_time in token_gen:
            token = self._token_to_id(token_text, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Convert to audio when we have enough tokens
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    
                    audio_start = time.time()
                    audio_samples = self._convert_to_audio(buffer_to_proc)
                    audio_time = time.time() - audio_start
                    
                    if audio_samples is not None:
                        audio_chunks_generated += 1
                        print(f"Audio chunk {audio_chunks_generated}: {audio_time:.3f}s to convert {len(buffer_to_proc)} tokens")
                        yield audio_samples
        
        print(f"Audio decoding complete: {audio_chunks_generated} audio chunks generated")

    def _convert_to_audio(self, multiframe: list[int]) -> np.ndarray | None:
        if len(multiframe) < 28:  # Ensure we have enough tokens
            return None

        num_frames = len(multiframe) // 7
        frame = multiframe[: num_frames * 7]

        # Initialize empty numpy arrays instead of torch tensors
        codes_0 = np.array([], dtype=np.int32)
        codes_1 = np.array([], dtype=np.int32)
        codes_2 = np.array([], dtype=np.int32)

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

    def _token_gen(self, text: str, voice: str = "tara") -> Generator[tuple[str, float], None, None]:
        text = f"<|audio|>{voice}: {text}<|eot_id|><custom_token_4>"
        completion_url = "http://localhost:1234/v1/completions"  # Our llama-server endpoint
        data = {
            "stream": True,
            "prompt": text,
            "max_tokens": 2048,
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
        
        for line in response.iter_lines():
            line = line.decode("utf-8")

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
                            yield token, token_time
                except json.JSONDecodeError:
                    continue
        
        total_token_time = time.time() - request_start
        print(f"Token generation complete: {token_count} tokens in {total_token_time:.3f}s ({token_count/total_token_time:.1f} tokens/sec)")

    def tts(self, text: str, voice: str = "tara"):
        overall_start = time.time()
        
        print(f"\n=== TTS Generation Started ===")
        print(f"Input text: '{text}' (voice: {voice})")
        
        buffer = []
        decode_start = time.time()
        
        for audio_bytes in self._decode(self._token_gen(text, voice)):
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

def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech with OrpheusCpp")
    parser.add_argument("--text", type=str, required=True, help="The text to convert to speech")
    parser.add_argument("--voice", type=str, choices=["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"], default="tara", help="The voice to use for the TTS")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file")
    args = parser.parse_args()

    orpheus = OrpheusCpp()
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
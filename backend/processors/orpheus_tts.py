"""
OrpheusTTS processor for text-to-speech generation.

This module provides:
- OrpheusTTS: Integrated Orpheus model (llama-cpp-python) and SNAC decoder
- Direct model inference without HTTP overhead
- Dual model warmup for optimal performance
"""

import time
import numpy as np
from pathlib import Path
from typing import Generator, Optional
from llama_cpp import Llama
from snac import SNAC
import torch
import sys
import os

# Import decoder functions
try:
    from ..decoder import turn_token_into_id, tokens_decoder_sync
    from ..decoder import convert_to_audio as _convert_to_audio_base
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from decoder import turn_token_into_id, tokens_decoder_sync
    from decoder import convert_to_audio as _convert_to_audio_base

# Helper to detect if running in Uvicorn's reloader
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

IS_RELOADER = is_reloader_process()

class OrpheusTTS:
    """Integrated OrpheusTTS with direct model inference"""
    
    def __init__(self):
        """Initialize OrpheusTTS with Orpheus and SNAC models"""
        
        # Test CUDA availability for GPU acceleration
        test_tensor = torch.zeros(10, device="cuda")
        del test_tensor
        print("✅ CUDA test successful - GPU acceleration enabled")
        
        # Load Orpheus model via llama-cpp-python with GPU acceleration
        model_path = Path(__file__).parent.parent.parent / "models" / "Orpheus-3b-FT-Q2_K.gguf"
        if not IS_RELOADER:
            print(f"🔧 Loading Orpheus model: {model_path.name} (GPU-accelerated)")
        
        self.orpheus_llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,          # Context window
            n_batch=512,         # Batch size  
            flash_attn=True,     # Enable Flash Attention
            n_gpu_layers=-1,     # Use all GPU layers for acceleration
        )
        
        # Load SNAC model with CUDA
        if not IS_RELOADER:
            print(f"🔧 Loading SNAC model with CUDA acceleration...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        
        # Set device for SNAC with CUDA acceleration
        self.snac_device = "cuda"
        if not IS_RELOADER:
            print(f"🔧 SNAC device: {self.snac_device}")
        
        self.snac_model = self.snac_model.to(self.snac_device)
        
        # Dual model warmup
        self._warmup_models()
        
    def _warmup_models(self):
        """Warmup both Orpheus and SNAC models"""
        if not IS_RELOADER:
            print("🔥 Starting dual model warmup (GPU-accelerated)...")
            
            # Orpheus warmup
            print("🔥 Warming up Orpheus model...")
            orpheus_start = time.time()
            warmup_response = self.orpheus_llm("<|audio|>tara: Hello<|eot_id|>", max_tokens=10)
            orpheus_warmup_time = (time.time() - orpheus_start) * 1000
            print(f"🔥 Orpheus warmup: {orpheus_warmup_time:.0f}ms")
            
            # SNAC warmup
            print("🔥 Warming up SNAC model...")
            snac_start = time.time()
            dummy_tokens = [1000, 1001, 1002, 1003, 1004, 1005, 1006] * 4  # 28 tokens
            self._convert_to_audio(dummy_tokens, 28)  # Warmup SNAC processing
            snac_warmup_time = (time.time() - snac_start) * 1000
            print(f"🔥 SNAC warmup: {snac_warmup_time:.0f}ms")
            
            total_warmup = orpheus_warmup_time + snac_warmup_time
            print(f"🔥 Total warmup time: {total_warmup:.0f}ms (GPU-accelerated)")
            print(f"✅ OrpheusTTS initialization complete - ready for high-performance inference")
    
    def _convert_to_audio(self, multiframe, count):
        """Convert tokens to audio using the integrated SNAC model"""
        # Temporarily set global variables for compatibility with existing decoder functions
        try:
            import backend.decoder as decoder_module
        except ImportError:
            import decoder as decoder_module
            
        original_model = decoder_module.model
        original_device = decoder_module.snac_device
        
        # Set our model and device in the decoder module
        decoder_module.model = self.snac_model
        decoder_module.snac_device = self.snac_device
        
        # Call the base conversion function
        result = _convert_to_audio_base(multiframe, count)
        
        # Restore original values
        decoder_module.model = original_model
        decoder_module.snac_device = original_device
        
        return result
    
    def _tokens_decoder_sync(self, syn_token_gen):
        """Custom tokens decoder that uses our integrated SNAC model"""
        import asyncio
        import threading
        import queue
        
        # Use a larger queue for RTX 4090 to maximize GPU utilization
        max_queue_size = 32 if self.snac_device == "cuda" else 8
        audio_queue = queue.Queue(maxsize=max_queue_size)
        
        # Collect tokens in batches for higher throughput
        batch_size = 16 if self.snac_device == "cuda" else 4
        
        # Convert the synchronous token generator into an async generator with batching
        async def async_token_gen():
            token_batch = []
            for token in syn_token_gen:
                token_batch.append(token)
                # Process in batches for efficiency
                if len(token_batch) >= batch_size:
                    for t in token_batch:
                        yield t
                    token_batch = []
            # Process any remaining tokens
            for t in token_batch:
                yield t

        async def async_producer():
            # Start timer for performance logging
            start_time = time.time()
            chunk_count = 0
            
            try:
                # Process audio chunks from the token decoder
                async for audio_chunk in self._tokens_decoder(async_token_gen()):
                    if audio_chunk:  # Validate audio chunk before adding to queue
                        audio_queue.put(audio_chunk)
                        chunk_count += 1
                        
                        # Log performance stats periodically
                        if chunk_count % 10 == 0:
                            elapsed = time.time() - start_time
                            print(f"Generated {chunk_count} chunks in {elapsed:.2f}s ({chunk_count/elapsed:.2f} chunks/sec)")
            except Exception as e:
                print(f"Error in audio producer: {e}")
                import traceback
                traceback.print_exc()
            finally:    
                # Signal completion
                print("Audio producer completed - finalizing all chunks")
                audio_queue.put(None)  # Sentinel

        def run_async():
            asyncio.run(async_producer())

        # Use a higher priority thread for RTX 4090 to ensure it stays fed with work
        thread = threading.Thread(target=run_async)
        thread.daemon = True  # Allow the thread to be terminated when the main thread exits
        thread.start()

        # Yield chunks immediately as they arrive (no buffering for true streaming)
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            
            # Immediate yield - no buffering or grouping
            yield audio

        thread.join()
    
    async def _tokens_decoder(self, token_gen):
        """Custom async token decoder using integrated SNAC model"""
        buffer = []
        count = 0
        
        async for token_sim in token_gen:       
            token = turn_token_into_id(token_sim, count)
            if token is not None and token > 0:
                buffer.append(token)
                count += 1

                # Original Orpheus logic: process every 7 tokens after count > 27
                if count % 7 == 0 and count > 27:
                    buffer_to_proc = buffer[-28:]
                    audio_samples = self._convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
        
        # CRITICAL: End-of-generation handling - process all remaining frames
        # Process remaining complete frames (ideal size)
        if len(buffer) >= 49:
            buffer_to_proc = buffer[-49:]
            audio_samples = self._convert_to_audio(buffer_to_proc, count)
            if audio_samples is not None:
                yield audio_samples
                
        # Process any additional complete frames (minimum size)
        elif len(buffer) >= 28:
            buffer_to_proc = buffer[-28:]
            audio_samples = self._convert_to_audio(buffer_to_proc, count)
            if audio_samples is not None:
                yield audio_samples
                
        # Final special case: even if we don't have minimum frames, try to process
        # what we have by padding with silence tokens that won't affect the audio
        elif len(buffer) >= 7:
            # Pad to minimum frame requirement with copies of the final token
            # This is more continuous than using unrelated tokens from the beginning
            last_token = buffer[-1]
            padding_needed = 28 - len(buffer)
            
            # Create a padding array of copies of the last token
            # This maintains continuity much better than circular buffering
            padding = [last_token] * padding_needed
            padded_buffer = buffer + padding
            
            print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} repeated-token padding")
            audio_samples = self._convert_to_audio(padded_buffer, count)
            if audio_samples is not None:
                yield audio_samples
    
    def _generate_tokens_stream(self, text: str, voice: str = "tara") -> Generator[str, None, None]:
        """Generate tokens using direct Orpheus model inference"""
        text_formatted = f"<|audio|>{voice}: {text}<|eot_id|>"
        
        print(f"Starting token generation for: '{text_formatted}'")
        request_start = time.time()
        
                
        # Create streaming completion with correct llama-cpp-python parameters
        # Use logit bias to suppress end tokens and force shorter generation
        stream = self.orpheus_llm.create_completion(
            text_formatted,
            max_tokens=2000,            # Use smaller max_tokens to prevent over-generation
            stream=True,
            temperature=0.6,           # Match inference.py setting
            top_k=1,
            top_p=0.9,                 # Match inference.py setting
            repeat_penalty=1.1,        # Critical for quality output per inference.py
        )
        
        first_token_time = None
        token_count = 0
        last_token_time = request_start
        
        for output in stream:
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                token = choice.get('text', '')
                
                # Note: Token IDs not available in streaming response, using logit bias instead
                
                if token:
                    token_time = time.time()
                    if first_token_time is None:
                        first_token_time = token_time
                        time_to_first_token = first_token_time - request_start
                        print(f"Time to first token: {time_to_first_token:.3f}s")
                    
                    token_count += 1
                    time_between_tokens = token_time - last_token_time
                    
                    if token_count <= 5 or token_count % 10 == 0:
                        print(f"Token {token_count}: received in {time_between_tokens:.3f}s since last token")
                    
                    last_token_time = token_time
                    yield token
        
        total_token_time = time.time() - request_start
        print(f"Token generation complete: {token_count} tokens in {total_token_time:.3f}s ({token_count/total_token_time:.1f} tokens/sec)")
    
    def tts_streaming(self, text: str, voice: str = "tara", save_chunks: bool = False, chunk_prefix: str = "chunk"):
        """
        Generate TTS audio with streaming support using integrated models.
        
        Args:
            text: The text to convert to speech
            voice: The voice to use (default: "tara")
            save_chunks: Whether to save individual chunks to disk
            chunk_prefix: Prefix for chunk filenames
            
        Yields:
            Individual chunks: (sample_rate, audio_array, chunk_count)
            Final result: (sample_rate, audio_data, total_time, audio_duration, is_final)
        """
        import soundfile
        
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
            for i, token in enumerate(self._generate_tokens_stream(text, voice)):
                if i == 0 and first_token_time is None:
                    first_token_time = time.time() - token_gen_start
                yield token
        
        decode_start = time.time()
        
        for audio_bytes in self._tokens_decoder_sync(token_gen_with_timing()):
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
import numpy as np
import torch
import asyncio
import threading
import queue
import time

def convert_to_audio(multiframe, count, model, device, cuda_stream=None):
    """
    Convert tokens to audio using the provided SNAC model.
    
    Args:
        multiframe: List of token IDs to convert
        count: Current token count (unused but kept for compatibility)
        model: SNAC model instance to use for decoding
        device: Device to run the model on ('cuda', 'cpu', etc.)
        cuda_stream: Optional CUDA stream for parallel processing
    
    Returns:
        bytes: Audio data as int16 bytes, or None if invalid input
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=device)
    
    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=device)
    
    # Direct indexing
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    
    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)
        
        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process on GPU if possible, with minimal data transfer
        if device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
    return audio_bytes

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    This is the definitive implementation used by both inference.py and speechpipe.py.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
    # Check cache first (significant speedup for repeated tokens)
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
        
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
        
    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
        
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
            
        return token_id
    except (ValueError, IndexError):
        return None

async def tokens_decoder(token_gen, model, device, cuda_stream=None):
    """
    Async token decoder that converts tokens to audio chunks.
    
    Args:
        token_gen: Async generator yielding tokens
        model: SNAC model instance to use for decoding
        device: Device to run the model on
        cuda_stream: Optional CUDA stream for parallel processing
    """
    buffer = []
    count = 0
    last_processed_index = 0  # Track the last token index we processed
    
    async for token_sim in token_gen:       
        token = turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Original Orpheus logic: process every 7 tokens after count > 27
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count, model, device, cuda_stream)
                if audio_samples is not None:
                    yield audio_samples
                    last_processed_index = len(buffer)  # Mark that we've processed up to here
    
    # CRITICAL: End-of-generation handling - only process unprocessed tokens
    # Calculate how many tokens remain unprocessed
    remaining_tokens = len(buffer) - last_processed_index
    
    # Only process if we have unprocessed tokens beyond what was already handled
    if remaining_tokens >= 7:
        # Get only the unprocessed portion of the buffer
        unprocessed_buffer = buffer[last_processed_index:]
        
        # Process based on size of unprocessed portion
        if len(unprocessed_buffer) >= 28:
            # We have enough for a full frame
            buffer_to_proc = unprocessed_buffer[:28]  # Take first 28 of unprocessed
            audio_samples = convert_to_audio(buffer_to_proc, count, model, device, cuda_stream)
            if audio_samples is not None:
                yield audio_samples
        else:
            # Need to pad the unprocessed tokens
            last_token = unprocessed_buffer[-1]
            padding_needed = 28 - len(unprocessed_buffer)
            padding = [last_token] * padding_needed
            padded_buffer = unprocessed_buffer + padding
            
            print(f"Processing final partial frame: {len(unprocessed_buffer)} unprocessed tokens + {padding_needed} repeated-token padding")
            audio_samples = convert_to_audio(padded_buffer, count, model, device, cuda_stream)
            if audio_samples is not None:
                yield audio_samples

# ------------------ Synchronous Tokens Decoder Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen, model, device, cuda_stream=None):
    """
    Synchronous wrapper for the async token decoder.
    
    Args:
        syn_token_gen: Synchronous generator yielding tokens
        model: SNAC model instance to use for decoding
        device: Device to run the model on
        cuda_stream: Optional CUDA stream for parallel processing
    
    Yields:
        Audio chunks as bytes
    """
    # Use a larger queue for RTX 4090 to maximize GPU utilization
    max_queue_size = 32 if device == "cuda" else 8
    audio_queue = queue.Queue(maxsize=max_queue_size)
    
    # Collect tokens in batches for higher throughput
    batch_size = 16 if device == "cuda" else 4
    
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
            async for audio_chunk in tokens_decoder(async_token_gen(), model, device, cuda_stream):
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
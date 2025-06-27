from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import os
import time

# Load pre-trained SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Determine device (CUDA or CPU)
snac_device = os.environ.get("SNAC_DEVICE", "cuda")
model = model.to(snac_device)

# Enable PyTorch optimizations
print("Applying PyTorch optimizations...")

# Enable CUDA optimizations if available
if torch.cuda.is_available():
    print("  ✅ CUDA optimizations enabled")
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable optimized attention mechanisms
    torch.backends.cuda.enable_flash_sdp(True)
    # Optimize memory allocator
    torch.cuda.empty_cache()

# Apply torch.compile for significant speedup (PyTorch 2.0+)
try:
    # Use different modes based on environment variable
    compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")  # options: default, reduce-overhead, max-autotune
    model = torch.compile(model, mode=compile_mode)
    print(f"  ✅ torch.compile enabled with mode: {compile_mode}")
except Exception as e:
    print(f"  ⚠️  torch.compile not available: {e}")

# Enable mixed precision if requested
use_mixed_precision = os.environ.get("SNAC_MIXED_PRECISION", "false").lower() == "true"
if use_mixed_precision:
    try:
        model = model.half()  # Convert to float16
        print("  ✅ Mixed precision (float16) enabled")
    except Exception as e:
        print(f"  ⚠️  Mixed precision failed: {e}")

print("PyTorch optimizations complete!")

def warmup_model():
    """Warmup the SNAC model to eliminate cold start penalty."""
    print("Warming up SNAC model...")
    warmup_start = time.time()
    
    # Create dummy inputs for warmup using the same dtype as inference
    warmup_dtype = torch.float16 if use_mixed_precision else torch.int32
    dummy_codes_0 = torch.tensor([[1, 2, 3, 4]], device=snac_device, dtype=warmup_dtype)
    dummy_codes_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=snac_device, dtype=warmup_dtype)
    dummy_codes_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], device=snac_device, dtype=warmup_dtype)
    
    codes = [dummy_codes_0, dummy_codes_1, dummy_codes_2]
    
    try:
        with torch.inference_mode():
            _ = model.decode(codes)
        warmup_time = time.time() - warmup_start
        print(f"Model warmup completed in {warmup_time:.3f}s")
    except Exception as e:
        print(f"Warmup failed: {e} - proceeding anyway")

# Perform warmup on module import
warmup_model()

def convert_to_audio(multiframe, count):
    """Convert multi-frame token data into audio bytes."""
    conversion_start = time.time()
    
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    # Time tensor preparation
    tensor_prep_start = time.time()
    
    # Orpheus-FastAPI optimization: Pre-allocate tensors directly on device
    dtype = torch.float16 if use_mixed_precision else torch.int32

	 # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    
    # Pre-allocate tensors with exact sizes (no numpy, no concatenation)
    codes_0 = torch.zeros(num_frames, dtype=dtype, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=dtype, device=snac_device)  
    codes_2 = torch.zeros(num_frames * 4, dtype=dtype, device=snac_device)
    
    # Direct tensor population - vectorized without numpy
    for i in range(num_frames):
        frame_offset = i * 7
        
        # codes_0: column 0
        codes_0[i] = frame_tensor[frame_offset]
        
        # codes_1: columns 1 and 4 
        codes_1[i * 2] = frame_tensor[frame_offset + 1]
        codes_1[i * 2 + 1] = frame_tensor[frame_offset + 4]
        
        # codes_2: columns 2, 3, 5, 6
        codes_2[i * 4] = frame_tensor[frame_offset + 2]
        codes_2[i * 4 + 1] = frame_tensor[frame_offset + 3] 
        codes_2[i * 4 + 2] = frame_tensor[frame_offset + 5]
        codes_2[i * 4 + 3] = frame_tensor[frame_offset + 6]
    
    # Add batch dimension and create codes list
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    # Fast validation: check ranges directly without dtype conversion
    if dtype == torch.int32:
        # Skip validation if using int32 (already validated by token parsing)
        valid = True
    else:
        # Only validate if using mixed precision
        valid = (torch.all(codes_0 >= 0) and torch.all(codes_0 <= 4096) and
                torch.all(codes_1 >= 0) and torch.all(codes_1 <= 4096) and
                torch.all(codes_2 >= 0) and torch.all(codes_2 <= 4096))
    
    if not valid:
        return None

    tensor_prep_end = time.time()
    tensor_prep_time = tensor_prep_end - tensor_prep_start

    # Time pure SNAC model inference
    model_inference_start = time.time()
    with torch.inference_mode():
        audio_hat = model.decode(codes)
    model_inference_end = time.time()
    model_inference_time = model_inference_end - model_inference_start

    # Time post-processing
    postprocess_start = time.time()
    audio_slice = audio_hat[:, :, 2048:4096]
    detached_audio = audio_slice.detach().cpu()
    audio_np = detached_audio.numpy()
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()
    postprocess_end = time.time()
    postprocess_time = postprocess_end - postprocess_start

    total_conversion_time = postprocess_end - conversion_start

    # Log detailed SNAC timing (only for first few chunks to avoid spam)
    if hasattr(convert_to_audio, 'call_count'):
        convert_to_audio.call_count += 1
    else:
        convert_to_audio.call_count = 1

    if convert_to_audio.call_count <= 3:
        print(f"  🔧 SNAC BREAKDOWN (call {convert_to_audio.call_count}):")
        print(f"    Tensor preparation: {tensor_prep_time:.3f}s")
        print(f"    Model inference: {model_inference_time:.3f}s (pure SNAC)")
        print(f"    Post-processing: {postprocess_time:.3f}s")
        print(f"    Total SNAC time: {total_conversion_time:.3f}s")
    elif convert_to_audio.call_count % 10 == 0:
        print(f"  🔧 SNAC call {convert_to_audio.call_count}: total={total_conversion_time:.3f}s, model={model_inference_time:.3f}s")

    return audio_bytes

# Token ID cache for performance optimization
token_id_cache = {}
cache_stats = {'hits': 0, 'misses': 0}
MAX_CACHE_SIZE = 10000

def turn_token_into_id(token_string, index):
    """Parse custom tokens from strings and extract numeric IDs with caching."""
    # Check cache first for speedup (Orpheus-FastAPI optimization)
    cache_key = (token_string.strip(), index % 7)
    if cache_key in token_id_cache:
        cache_stats['hits'] += 1
        return token_id_cache[cache_key]
    
    cache_stats['misses'] += 1
    
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        # Cache negative results too
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            
            # Cache the successful result
            if len(token_id_cache) < MAX_CACHE_SIZE:
                token_id_cache[cache_key] = token_id
            
            return token_id
        except ValueError:
            # Cache failed conversions too
            if len(token_id_cache) < MAX_CACHE_SIZE:
                token_id_cache[cache_key] = None
            return None
    else:
        # Cache negative results
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None

def get_cache_stats():
    """Get token cache performance statistics."""
    total = cache_stats['hits'] + cache_stats['misses']
    hit_rate = cache_stats['hits'] / total * 100 if total > 0 else 0
    return {
        'cache_size': len(token_id_cache),
        'max_size': MAX_CACHE_SIZE,
        'hits': cache_stats['hits'],
        'misses': cache_stats['misses'],
        'hit_rate': hit_rate
    }

async def tokens_decoder(token_gen):
    """Asynchronous generator that processes streaming tokens."""
    buffer = []
    count = 0
    last_token_time = None
    first_audio_generated = False
    chunk_count = 0
    
    print("🎵 Starting token decoder...")
    
    async for token_text, token_time in token_gen:
        # Track token timing
        if last_token_time is not None:
            token_interval = token_time - last_token_time
            if token_interval > 0.1:  # Log significant gaps
                print(f"⏱️  Token gap: {token_interval:.3f}s (waiting for server)")
        
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                tokens_ready_time = time.time()
                buffer_to_proc = buffer[-28:]
                
                # Time the SNAC conversion
                snac_start = time.time()
                audio_samples = convert_to_audio(buffer_to_proc, count)
                snac_end = time.time()
                snac_time = snac_end - snac_start
                
                if audio_samples is not None:
                    chunk_count += 1
                    
                    # Calculate timing breakdown
                    if last_token_time:
                        token_accumulation_time = tokens_ready_time - last_token_time
                    else:
                        token_accumulation_time = 0
                    
                    # Log detailed timing for first few chunks
                    if chunk_count <= 3:
                        print(f"🎵 Audio chunk {chunk_count} generated:")
                        print(f"  📊 TIMING BREAKDOWN:")
                        print(f"    Token accumulation: {token_accumulation_time:.3f}s (server dependency)")
                        print(f"    SNAC inference: {snac_time:.3f}s (model bottleneck)")
                        print(f"    Total chunk time: {(snac_end - (last_token_time or tokens_ready_time)):.3f}s")
                        if not first_audio_generated:
                            print(f"  🚀 FIRST AUDIO CHUNK COMPLETE")
                            first_audio_generated = True
                    elif chunk_count % 5 == 0:
                        print(f"🎵 Chunk {chunk_count}: token_wait={token_accumulation_time:.3f}s, snac={snac_time:.3f}s")
                    
                    yield audio_samples
        
        last_token_time = token_time
    
    # Report cache performance
    stats = get_cache_stats()
    print(f"🎵 Token decoder complete: {chunk_count} audio chunks generated")
    print(f"📊 Token cache stats: {stats['hit_rate']:.1f}% hit rate ({stats['hits']} hits, {stats['misses']} misses, {stats['cache_size']} cached)")

def tokens_decoder_sync(syn_token_gen):
    """Synchronous wrapper for the async token decoder."""
    result_queue = queue.Queue()
    sync_start = time.time()
    chunk_count = 0
    
    async def async_wrapper():
        try:
            async for audio_chunk in tokens_decoder(syn_token_gen):
                chunk_put_time = time.time()
                result_queue.put(('chunk', audio_chunk, chunk_put_time))
        except Exception as e:
            result_queue.put(('error', e, time.time()))
        finally:
            result_queue.put(('done', None, time.time()))
    
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(async_wrapper())
        finally:
            loop.close()
    
    # Start async processing in a separate thread
    thread = threading.Thread(target=run_async)
    thread.start()
    
    print("🔄 Starting synchronous token decoder wrapper...")
    
    # Yield results as they become available
    while True:
        try:
            result = result_queue.get(timeout=10)
            queue_get_time = time.time()
            
            if len(result) >= 3:
                msg_type, data, put_time = result[0], result[1], result[2]
                queue_delay = queue_get_time - put_time
                
                if msg_type == 'chunk':
                    chunk_count += 1
                    
                    # Log queue delays for first few chunks
                    if chunk_count <= 3:
                        print(f"  📦 Queue timing (chunk {chunk_count}):")
                        print(f"    Queue delay: {queue_delay:.6f}s (threading overhead)")
                    elif queue_delay > 0.01:  # Only log significant delays
                        print(f"  📦 Queue delay chunk {chunk_count}: {queue_delay:.6f}s")
                    
                    yield data
                elif msg_type == 'error':
                    raise data
                elif msg_type == 'done':
                    total_sync_time = queue_get_time - sync_start
                    print(f"🔄 Sync wrapper complete: {chunk_count} chunks in {total_sync_time:.3f}s")
                    break
            else:
                # Handle old format without timing
                msg_type, data = result[0], result[1]
                if msg_type == 'chunk':
                    yield data
                elif msg_type == 'error':
                    raise data
                elif msg_type == 'done':
                    break
                    
        except queue.Empty:
            print("⚠️  Queue timeout - no data received")
            break
    
    thread.join()
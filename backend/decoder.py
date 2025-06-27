"""
Optimized SNAC decoder based on Orpheus-FastAPI implementation
Provides high-performance audio generation from token streams
"""

from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys

def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

IS_RELOADER = is_reloader_process()

# Constants
CUSTOM_TOKEN_PREFIX = "<custom_token_"
MAX_CACHE_SIZE = 10000

# PyTorch and CUDA optimization checks
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
        if not IS_RELOADER:
            print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        if not IS_RELOADER:
            print("CUDA graphs support is available")
except:
    pass

# Device selection with priority: CUDA > MPS > CPU
snac_device = ("cuda" if torch.cuda.is_available() 
               else "mps" if torch.backends.mps.is_available() 
               else "cpu")

if not IS_RELOADER:
    print(f"Using device: {snac_device}")

# Load SNAC model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
model = model.to(snac_device)

# CUDA optimizations
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")
    
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable optimized attention mechanisms
    torch.backends.cuda.enable_flash_sdp(True)

# Apply torch.compile for significant speedup (PyTorch 2.0+)
if TORCH_COMPILE_AVAILABLE:
    try:
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", "default")
        model = torch.compile(model, mode=compile_mode)
        if not IS_RELOADER:
            print(f"torch.compile enabled with mode: {compile_mode}")
    except Exception as e:
        if not IS_RELOADER:
            print(f"torch.compile failed: {e}")

# Mixed precision support
use_mixed_precision = os.environ.get("SNAC_MIXED_PRECISION", "false").lower() == "true"
if use_mixed_precision and snac_device == "cuda":
    try:
        model = model.half()
        if not IS_RELOADER:
            print("Mixed precision (float16) enabled")
    except Exception as e:
        if not IS_RELOADER:
            print(f"Mixed precision failed: {e}")
        use_mixed_precision = False

# Token ID cache for performance optimization
token_id_cache = {}
cache_stats = {'hits': 0, 'misses': 0}

def turn_token_into_id(token_string, index):
    """Parse custom tokens from strings and extract numeric IDs with caching."""
    # Check cache first for speedup
    cache_key = (token_string.strip(), index % 7)
    if cache_key in token_id_cache:
        cache_stats['hits'] += 1
        return token_id_cache[cache_key]
    
    cache_stats['misses'] += 1
    
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith('>')):
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None
    
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
        
        return token_id
    except (ValueError, IndexError):
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = None
        return None

def convert_to_audio(multiframe, count):
    """Optimized audio conversion with reduced CPU-GPU transfers"""
    conversion_start = time.time()
    
    if len(multiframe) < 7:
        return None
    
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]
    
    # Time tensor preparation
    tensor_prep_start = time.time()
    
    # Determine dtype based on mixed precision setting
    dtype = torch.float16 if use_mixed_precision else torch.int32
    
    # Pre-allocate tensors directly on device for optimal performance
    codes_0 = torch.zeros(num_frames, dtype=dtype, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=dtype, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=dtype, device=snac_device)
    
    # Vectorized tensor population - much faster than loops
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    frame_reshaped = frame_tensor.view(num_frames, 7)
    
    # Extract codes using advanced indexing (vectorized)
    codes_0.copy_(frame_reshaped[:, 0].to(dtype))
    codes_1[0::2] = frame_reshaped[:, 1].to(dtype)
    codes_1[1::2] = frame_reshaped[:, 4].to(dtype)
    codes_2[0::4] = frame_reshaped[:, 2].to(dtype)
    codes_2[1::4] = frame_reshaped[:, 3].to(dtype)
    codes_2[2::4] = frame_reshaped[:, 5].to(dtype)
    codes_2[3::4] = frame_reshaped[:, 6].to(dtype)
    
    # Create codes list with batch dimension
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
    
    # Fast validation (skip for int32 as it's already validated)
    if dtype != torch.int32:
        if not (torch.all(codes_0 >= 0) and torch.all(codes_0 <= 4096) and
                torch.all(codes_1 >= 0) and torch.all(codes_1 <= 4096) and
                torch.all(codes_2 >= 0) and torch.all(codes_2 <= 4096)):
            return None
    
    tensor_prep_end = time.time()
    tensor_prep_time = tensor_prep_end - tensor_prep_start
    
    # Time pure SNAC model inference
    model_inference_start = time.time()
    
    # Use CUDA stream for parallel processing if available
    if cuda_stream is not None:
        with torch.cuda.stream(cuda_stream):
            with torch.inference_mode():
                audio_hat = model.decode(codes)
    else:
        with torch.inference_mode():
            audio_hat = model.decode(codes)
    
    model_inference_end = time.time()
    model_inference_time = model_inference_end - model_inference_start
    
    # Time post-processing
    postprocess_start = time.time()
    audio_slice = audio_hat[:, :, 2048:4096]
    
    # Optimized scaling for different devices
    if snac_device == "cuda":
        # Keep on GPU longer for RTX cards
        audio_np = (audio_slice * 32767).detach().cpu().numpy().astype(np.int16)
    else:
        # Standard CPU processing
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_np = (audio_np * 32767).astype(np.int16)
    
    audio_bytes = audio_np.tobytes()
    postprocess_end = time.time()
    postprocess_time = postprocess_end - postprocess_start
    
    total_conversion_time = postprocess_end - conversion_start
    
    # Log detailed timing for performance analysis
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

def warmup_model():
    """Warmup the SNAC model to eliminate cold start penalty."""
    if IS_RELOADER:
        return
        
    print("Warming up SNAC model...")
    warmup_start = time.time()
    
    # Create dummy inputs for warmup using the same dtype as inference
    warmup_dtype = torch.float16 if use_mixed_precision else torch.int32
    dummy_codes_0 = torch.tensor([[1, 2, 3, 4]], device=snac_device, dtype=warmup_dtype)
    dummy_codes_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=snac_device, dtype=warmup_dtype)
    dummy_codes_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], device=snac_device, dtype=warmup_dtype)
    
    codes = [dummy_codes_0, dummy_codes_1, dummy_codes_2]
    
    try:
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                with torch.inference_mode():
                    _ = model.decode(codes)
        else:
            with torch.inference_mode():
                _ = model.decode(codes)
        warmup_time = time.time() - warmup_start
        print(f"Model warmup completed in {warmup_time:.3f}s")
    except Exception as e:
        print(f"Warmup failed: {e} - proceeding anyway")

async def tokens_decoder(token_gen):
    """Asynchronous generator that processes streaming tokens with optimized performance."""
    buffer = []
    count = 0
    last_token_time = None
    first_audio_generated = False
    chunk_count = 0
    
    print("🎵 Starting optimized token decoder...")
    
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
    """Synchronous wrapper for the async token decoder with performance monitoring."""
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
    
    print("🔄 Starting optimized synchronous token decoder wrapper...")
    
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

# Perform warmup on module import
warmup_model()

if not IS_RELOADER:
    print("✅ Optimized decoder ready!")
    print(f"  Device: {snac_device}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"  Torch compile: {TORCH_COMPILE_AVAILABLE}")
    print(f"  CUDA graphs: {CUDA_GRAPHS_AVAILABLE}")
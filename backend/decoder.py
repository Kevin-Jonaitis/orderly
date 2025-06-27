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

def warmup_model():
    """Warmup the SNAC model to eliminate cold start penalty."""
    print("Warming up SNAC model...")
    warmup_start = time.time()
    
    # Create dummy inputs for warmup
    dummy_codes_0 = torch.tensor([[1, 2, 3, 4]], device=snac_device, dtype=torch.int32)
    dummy_codes_1 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=snac_device, dtype=torch.int32)
    dummy_codes_2 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]], device=snac_device, dtype=torch.int32)
    
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
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames * 7]

    codes_0 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_1 = torch.tensor([], device=snac_device, dtype=torch.int32)
    codes_2 = torch.tensor([], device=snac_device, dtype=torch.int32)

    for j in range(num_frames):
        i = 7 * j
        # Create tensors for each code segment
        code_0 = torch.tensor([frame[i]], device=snac_device, dtype=torch.int32)
        code_1 = torch.tensor([frame[i + 1], frame[i + 4]], device=snac_device, dtype=torch.int32)
        code_2 = torch.tensor([frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]], device=snac_device, dtype=torch.int32)
        
        codes_0 = torch.cat([codes_0, code_0])
        codes_1 = torch.cat([codes_1, code_1])
        codes_2 = torch.cat([codes_2, code_2])

    # Add batch dimension and create codes list
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    # Validate token ranges
    if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or \
       torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or \
       torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
        return None

    with torch.inference_mode():
        audio_hat = model.decode(codes)
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes

def turn_token_into_id(token_string, index):
    """Parse custom tokens from strings and extract numeric IDs."""
    token_string = token_string.strip()
    
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

async def tokens_decoder(token_gen):
    """Asynchronous generator that processes streaming tokens."""
    buffer = []
    count = 0
    
    async for token_text, token_time in token_gen:
        token = turn_token_into_id(token_text, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            # Convert to audio when we have enough tokens
            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples

def tokens_decoder_sync(syn_token_gen):
    """Synchronous wrapper for the async token decoder."""
    result_queue = queue.Queue()
    
    async def async_wrapper():
        try:
            async for audio_chunk in tokens_decoder(syn_token_gen):
                result_queue.put(('chunk', audio_chunk))
        except Exception as e:
            result_queue.put(('error', e))
        finally:
            result_queue.put(('done', None))
    
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
    
    # Yield results as they become available
    while True:
        try:
            msg_type, data = result_queue.get(timeout=10)
            if msg_type == 'chunk':
                yield data
            elif msg_type == 'error':
                raise data
            elif msg_type == 'done':
                break
        except queue.Empty:
            break
    
    thread.join()
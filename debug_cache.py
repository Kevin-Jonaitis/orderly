#!/usr/bin/env python3
"""
Debug cache behavior to understand why performance is worse
"""

import time
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from llama_cpp import Llama, LlamaRAMCache

def debug_cache_behavior():
    """Debug what's happening with the cache"""
    
    model_path = Path(__file__).parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
    
    # Create two identical prompts that should have cache reuse
    base_system = """<|system|>
You are an AI fast-food order taker. Respond to the input in a reasonable manner, with their updated order at the end of your response.
Here is the user's current order:
- Waffle

Here is the menu for items and pricing:
- Cheeseburger: $11.99
- Fries: $5.99
- Waffle: $4.20
- Coke: $1.00
- Lemonade: $1.50
<|end|>
<|user|>
"""
    
    prompt1 = base_system + "can i get a cheeseburger<|end|>\n<|assistant|>"
    prompt2 = base_system + "add fries please<|end|>\n<|assistant|>"
    
    print("🔍 Cache Debug Test")
    print("="*50)
    
    # Test cache behavior
    llm = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_gpu_layers=-1,
        n_batch=128,
        flash_attn=True,
        use_mlock=True,
        verbose=False
    )
    
    cache = LlamaRAMCache(capacity_bytes=512*1024*1024)
    llm.set_cache(cache)
    
    print(f"📊 Initial cache state: {len(cache.cache_state)} entries")
    
    # Tokenize and analyze prompts
    tokens1 = llm.tokenize(prompt1.encode())
    tokens2 = llm.tokenize(prompt2.encode())
    
    print(f"\n🧠 Prompt analysis:")
    print(f"Prompt 1 tokens: {len(tokens1)}")
    print(f"Prompt 2 tokens: {len(tokens2)}")
    
    # Find common prefix
    common_prefix_len = 0
    for t1, t2 in zip(tokens1, tokens2):
        if t1 == t2:
            common_prefix_len += 1
        else:
            break
    
    print(f"Common prefix tokens: {common_prefix_len} / {len(tokens1)} ({common_prefix_len/len(tokens1)*100:.1f}%)")
    
    # Test first prompt
    print(f"\n🚀 Testing first prompt...")
    start = time.time()
    response1 = llm(prompt1, max_tokens=50)
    time1 = time.time() - start
    print(f"Time: {time1*1000:.0f}ms")
    print(f"Cache entries after first: {len(cache.cache_state)}")
    
    # Inspect cache keys
    if hasattr(cache, 'cache_state'):
        for i, key in enumerate(cache.cache_state.keys()):
            print(f"  Cache key {i}: {len(key)} tokens")
    
    # Test second prompt
    print(f"\n🚀 Testing second prompt (should reuse system prompt)...")
    start = time.time()
    response2 = llm(prompt2, max_tokens=50)
    time2 = time.time() - start
    print(f"Time: {time2*1000:.0f}ms")
    print(f"Cache entries after second: {len(cache.cache_state)}")
    
    # Analyze results
    improvement = (time1 - time2) / time1 * 100
    print(f"\n📊 Results:")
    print(f"First request: {time1*1000:.0f}ms")
    print(f"Second request: {time2*1000:.0f}ms")
    print(f"Improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print(f"✅ Cache is working! {improvement:.1f}% faster on second request")
    else:
        print(f"❌ Cache not providing benefit ({abs(improvement):.1f}% slower)")
    
    # Test cache overhead
    print(f"\n🔄 Testing cache overhead...")
    
    # Disable cache and test same prompts
    llm.set_cache(None)
    
    print("Testing without cache...")
    start = time.time()
    _ = llm(prompt1, max_tokens=50)
    time_no_cache_1 = time.time() - start
    
    start = time.time()
    _ = llm(prompt2, max_tokens=50)
    time_no_cache_2 = time.time() - start
    
    print(f"Without cache - First: {time_no_cache_1*1000:.0f}ms, Second: {time_no_cache_2*1000:.0f}ms")
    
    cache_overhead = ((time1 + time2) - (time_no_cache_1 + time_no_cache_2)) / (time_no_cache_1 + time_no_cache_2) * 100
    print(f"Cache overhead: {cache_overhead:.1f}%")
    
    # Test with identical prompts
    print(f"\n🔄 Testing with identical prompts...")
    llm.set_cache(LlamaRAMCache(capacity_bytes=512*1024*1024))
    
    identical_prompt = base_system + "test message<|end|>\n<|assistant|>"
    
    start = time.time()
    _ = llm(identical_prompt, max_tokens=50)
    time_identical_1 = time.time() - start
    
    start = time.time()  
    _ = llm(identical_prompt, max_tokens=50)  # Exact same prompt
    time_identical_2 = time.time() - start
    
    identical_improvement = (time_identical_1 - time_identical_2) / time_identical_1 * 100
    print(f"Identical prompts - First: {time_identical_1*1000:.0f}ms, Second: {time_identical_2*1000:.0f}ms")
    print(f"Identical prompt improvement: {identical_improvement:.1f}%")

if __name__ == "__main__":
    debug_cache_behavior()
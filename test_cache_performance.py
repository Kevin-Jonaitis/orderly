#!/usr/bin/env python3
"""
Test cache performance for system prompt reuse
"""

import time
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from llama_cpp import Llama, LlamaRAMCache

def test_cache_performance():
    """Test performance with and without caching"""
    
    # Model configuration (matching current settings)
    model_path = Path(__file__).parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
    
    def build_phi3_prompt(user_input: str) -> str:
        """Build properly formatted Phi-3 prompt with system context"""
        return f"""<|system|>
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
{user_input}<|end|>
<|assistant|>"""

    # Different user inputs to test
    test_inputs = [
        "can i get a cheeseburger with fries",
        "actually make that a large fries", 
        "and add a coke please",
        "wait skip the coke, i want lemonade",
        "how much is my total?"
    ]
    
    print("🧪 Cache Performance Test for System Prompt Reuse")
    print("="*60)
    
    # Test 1: Without Cache
    print("\n📊 Testing WITHOUT cache (baseline)")
    print("-" * 40)
    
    llm_no_cache = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_gpu_layers=-1,
        n_batch=128,
        flash_attn=True,
        use_mlock=True,
        verbose=False
    )
    
    # Warmup
    print("Warming up...")
    _ = llm_no_cache(build_phi3_prompt("test"), max_tokens=10)
    
    no_cache_times = []
    no_cache_tokens = []
    
    for i, user_input in enumerate(test_inputs):
        prompt = build_phi3_prompt(user_input)
        print(f"\nRequest {i+1}: '{user_input}'")
        
        start = time.time()
        response = llm_no_cache(prompt, max_tokens=100)
        elapsed = time.time() - start
        
        tokens = response['usage']['completion_tokens']
        no_cache_times.append(elapsed)
        no_cache_tokens.append(tokens)
        
        print(f"  Time: {elapsed*1000:.0f}ms, Tokens: {tokens}, Rate: {tokens/elapsed:.1f} tok/s")
    
    avg_no_cache = sum(no_cache_times) / len(no_cache_times)
    print(f"\n📈 Average without cache: {avg_no_cache*1000:.0f}ms")
    
    # Clean up
    del llm_no_cache
    
    # Test 2: With Cache
    print("\n📊 Testing WITH cache (optimized)")
    print("-" * 40)
    
    llm_with_cache = Llama(
        model_path=str(model_path),
        n_ctx=1024,
        n_gpu_layers=-1,
        n_batch=128,
        flash_attn=True,
        use_mlock=True,
        verbose=False
    )
    
    # Enable cache
    cache = LlamaRAMCache(capacity_bytes=512*1024*1024)  # 512MB
    llm_with_cache.set_cache(cache)
    print("🗂️  Cache enabled: 512MB capacity")
    
    # Warmup
    print("Warming up...")
    _ = llm_with_cache(build_phi3_prompt("test"), max_tokens=10)
    
    cached_times = []
    cached_tokens = []
    
    for i, user_input in enumerate(test_inputs):
        prompt = build_phi3_prompt(user_input)
        print(f"\nRequest {i+1}: '{user_input}'")
        
        start = time.time()
        response = llm_with_cache(prompt, max_tokens=100)
        elapsed = time.time() - start
        
        tokens = response['usage']['completion_tokens']
        cached_times.append(elapsed)
        cached_tokens.append(tokens)
        
        # Check cache status
        cache_entries = len(cache.cache_state) if hasattr(cache, 'cache_state') else 'unknown'
        cache_hit_indicator = "🎯" if i > 0 else "💾"  # Expect cache hits after first request
        
        print(f"  Time: {elapsed*1000:.0f}ms, Tokens: {tokens}, Rate: {tokens/elapsed:.1f} tok/s {cache_hit_indicator}")
        print(f"  Cache entries: {cache_entries}")
    
    avg_cached = sum(cached_times) / len(cached_times)
    print(f"\n📈 Average with cache: {avg_cached*1000:.0f}ms")
    
    # Analysis
    print("\n" + "="*60)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*60)
    
    improvement = (avg_no_cache - avg_cached) / avg_no_cache * 100
    print(f"Average without cache: {avg_no_cache*1000:.0f}ms")
    print(f"Average with cache:    {avg_cached*1000:.0f}ms")
    print(f"Improvement:           {improvement:.1f}% faster")
    print(f"Time saved per request: {(avg_no_cache - avg_cached)*1000:.0f}ms")
    
    # Per-request analysis
    print(f"\n📋 Per-Request Breakdown:")
    print(f"{'Request':<20} | {'No Cache':<10} | {'With Cache':<12} | {'Speedup':<8}")
    print("-" * 56)
    
    for i, (user_input, no_cache_time, cached_time) in enumerate(zip(test_inputs, no_cache_times, cached_times)):
        speedup = (no_cache_time - cached_time) / no_cache_time * 100
        print(f"{user_input[:18]:<20} | {no_cache_time*1000:>8.0f}ms | {cached_time*1000:>10.0f}ms | {speedup:>6.1f}%")
    
    # Expected cache behavior
    print(f"\n🎯 Cache Analysis:")
    print(f"Expected: First request builds cache, subsequent requests reuse system prompt")
    print(f"Cache should recognize identical system prompt portion and only process new user input")
    
    if improvement > 0:
        print(f"✅ Cache working! {improvement:.1f}% performance improvement")
        if avg_cached * 1000 < 200:
            print(f"🎉 Sub-200ms target achieved: {avg_cached*1000:.0f}ms average!")
    else:
        print(f"❌ Cache not providing expected benefit")
    
    # Clean up
    del llm_with_cache

if __name__ == "__main__":
    test_cache_performance()
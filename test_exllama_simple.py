#!/usr/bin/env python3
"""
Simple ExLlamaV2 test to isolate performance issues
"""

import time
import torch
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

def test_exllama_simple():
    print("🔥 Simple ExLlamaV2 Performance Test")
    print("=" * 50)
    
    # Load model
    model_path = Path("models/exl2_model")
    print(f"Loading model from: {model_path}")
    
    # Initialize config
    config = ExLlamaV2Config()
    config.model_dir = str(model_path)
    config.prepare()
    
    # Initialize model
    model = ExLlamaV2(config)
    
    # Initialize tokenizer
    tokenizer = ExLlamaV2Tokenizer(config)
    
    # Initialize cache
    cache = ExLlamaV2Cache(model, lazy=True)
    
    # Load model
    print("Loading model...")
    load_start = time.time()
    model.load_autosplit(cache)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # Initialize generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
    
    # Setup simple sampling
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.7
    settings.top_k = 50
    settings.top_p = 0.9
    
    # Simple test prompt
    simple_prompt = "Hello, how are you today?"
    print(f"\nTesting with prompt: '{simple_prompt}'")
    
    # Test generation
    print("\nGenerating response...")
    
    start_time = time.time()
    
    # Encode input
    input_ids = tokenizer.encode(simple_prompt)
    print(f"Input tokens: {len(input_ids)}")
    
    # Generate
    generator.warmup()
    generator.begin_stream(input_ids, settings)
    
    response_tokens = []
    max_tokens = 50
    
    for i in range(max_tokens):
        chunk, eos, _ = generator.stream()
        if eos or not chunk:
            break
        response_tokens.append(chunk)
        print(chunk, end='', flush=True)
        
        # Check if we're taking too long
        elapsed = time.time() - start_time
        if elapsed > 5.0:
            print(f"\n⚠️  TIMEOUT: Generation took more than 5s, stopping...")
            break
    
    total_time = time.time() - start_time
    full_response = "".join(response_tokens)
    
    print(f"\n\n📊 Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Tokens generated: {len(response_tokens)}")
    print(f"   Speed: {len(response_tokens)/total_time:.1f} tokens/sec")
    print(f"   Response: '{full_response}'")
    
    if total_time > 5.0:
        print("❌ FAILED: Took more than 5 seconds")
        return False
    else:
        print("✅ SUCCESS: Sub-5s generation")
        return True

if __name__ == "__main__":
    test_exllama_simple()
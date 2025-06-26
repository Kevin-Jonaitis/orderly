#!/usr/bin/env python3
"""
Test different quantizations for LLM performance
"""

import time
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from llama_cpp import Llama

def test_quantizations():
    """Test various quantizations and measure performance"""
    # Use the same prompt for consistent testing
    test_prompt = """<|system|>
You are an AI fast-food order taker. Respond to the input in a reasonable manner, with their updated order at the end of your response.
Here is the user's current order:
- Waffle

Here is the menu for items and pricing:
- Cheeseburger: $11.99
- Fries: $5.99
- Waffle: $4.20
- Coke: $1.00
<|end|>
<|user|>
can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade<|end|>
<|assistant|>"""
    
    # Define models to test
    models = [
        {
            "name": "Q4 (original)",
            "path": "models/Phi-3-mini-4k-instruct-q4.gguf",
            "size": "2.2GB"
        },
        {
            "name": "Q4_K_M",
            "path": "models/phi-3-mini-4k-instruct-q4_k_m.gguf", 
            "size": "2.2GB"
        },
        {
            "name": "Q4_K_S", 
            "path": "models/phi-3-mini-4k-instruct-q4_k_s.gguf",
            "size": "2.0GB"
        },
        {
            "name": "Q3_K_S",
            "path": "models/Phi-3-mini-4k-instruct-q3_k_s.gguf",
            "size": "1.7GB"
        }
    ]
    
    results = []
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing {model['name']} ({model['size']})")
        print(f"{'='*60}")
        
        model_path = Path(__file__).parent / model['path']
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            continue
            
        try:
            # Load model with optimal settings
            print(f"Loading model: {model_path}")
            llm = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_gpu_layers=-1,
                n_batch=128,  # Optimal batch size from previous test
                flash_attn=True,
                use_mlock=True,
                verbose=False  # Reduce verbosity for cleaner output
            )
            
            # Warmup run (not counted)
            print("Warming up...")
            _ = llm(test_prompt, max_tokens=10)
            
            # Test runs
            times = []
            tokens_generated = []
            
            for run in range(3):
                start = time.time()
                output = llm(test_prompt, max_tokens=100)
                elapsed = time.time() - start
                times.append(elapsed)
                
                # Count actual tokens generated
                token_count = output['usage']['completion_tokens']
                tokens_generated.append(token_count)
                
                tokens_per_sec = token_count / elapsed
                print(f"Run {run+1}: {elapsed*1000:.0f}ms, {token_count} tokens, {tokens_per_sec:.1f} tok/s")
            
            # Calculate averages
            avg_time = sum(times) / len(times)
            avg_tokens = sum(tokens_generated) / len(tokens_generated)
            avg_tok_per_sec = avg_tokens / avg_time
            
            results.append({
                'name': model['name'],
                'size': model['size'],
                'avg_time_ms': avg_time * 1000,
                'avg_tokens': avg_tokens,
                'avg_tok_per_sec': avg_tok_per_sec
            })
            
            print(f"\nAverage: {avg_time*1000:.0f}ms, {avg_tokens:.0f} tokens, {avg_tok_per_sec:.1f} tok/s")
            
            # Clean up
            del llm
            
        except Exception as e:
            print(f"Error with {model['name']}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Quantization Performance Comparison")
    print(f"{'='*70}")
    print(f"{'Model':>12} | {'Size':>6} | {'Time (ms)':>10} | {'Tokens':>7} | {'Tok/s':>8}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}")
    
    for result in results:
        print(f"{result['name']:>12} | {result['size']:>6} | {result['avg_time_ms']:>10.0f} | {result['avg_tokens']:>7.0f} | {result['avg_tok_per_sec']:>8.1f}")
    
    # Find optimal
    if results:
        best = min(results, key=lambda x: x['avg_time_ms'])
        print(f"\n🏆 Best quantization: {best['name']} ({best['avg_time_ms']:.0f}ms, {best['size']})")
        
        # Performance comparison
        print(f"\n📊 Performance vs Current:")
        current_q4 = next((r for r in results if r['name'] == 'Q4 (original)'), None)
        if current_q4:
            for result in results:
                if result['name'] != 'Q4 (original)':
                    speedup = (current_q4['avg_time_ms'] - result['avg_time_ms']) / current_q4['avg_time_ms'] * 100
                    if speedup > 0:
                        print(f"   {result['name']}: {speedup:.1f}% faster")
                    else:
                        print(f"   {result['name']}: {abs(speedup):.1f}% slower")

if __name__ == "__main__":
    print("🧪 Quantization Performance Test for Phi-3 Mini")
    test_quantizations()
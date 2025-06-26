#!/usr/bin/env python3
"""
Test different batch sizes for LLM performance
"""

import time
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from llama_cpp import Llama

def test_batch_sizes():
    """Test various batch sizes and measure performance"""
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

    # Model path - using Q4 as it performed better
    model_path = Path(__file__).parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
    
    results = []
    
    for batch_size in [128, 256, 384, 512, 768, 1024]:
        print(f"\n{'='*50}")
        print(f"Testing n_batch={batch_size}")
        print(f"{'='*50}")
        
        try:
            # Load model with specific batch size
            llm = Llama(
                model_path=str(model_path),
                n_ctx=1024,
                n_gpu_layers=-1,
                n_batch=batch_size,
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
                'batch_size': batch_size,
                'avg_time_ms': avg_time * 1000,
                'avg_tokens': avg_tokens,
                'avg_tok_per_sec': avg_tok_per_sec
            })
            
            print(f"\nAverage: {avg_time*1000:.0f}ms, {avg_tokens:.0f} tokens, {avg_tok_per_sec:.1f} tok/s")
            
            # Clean up
            del llm
            
        except Exception as e:
            print(f"Error with batch_size={batch_size}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - Batch Size Performance")
    print(f"{'='*60}")
    print(f"{'Batch':>6} | {'Time (ms)':>10} | {'Tokens':>7} | {'Tok/s':>8}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*8}")
    
    for result in results:
        print(f"{result['batch_size']:>6} | {result['avg_time_ms']:>10.0f} | {result['avg_tokens']:>7.0f} | {result['avg_tok_per_sec']:>8.1f}")
    
    # Find optimal
    best = min(results, key=lambda x: x['avg_time_ms'])
    print(f"\n🏆 Best batch size: {best['batch_size']} ({best['avg_time_ms']:.0f}ms)")

if __name__ == "__main__":
    print("🧪 Batch Size Performance Test for Phi-3 Mini")
    test_batch_sizes()
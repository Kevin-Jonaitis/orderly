#!/usr/bin/env python3
"""
Performance test for llama-cpp-python only (since ExLlamaV2 requires newer CUDA support)
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.processors.llm import LLMReasoner

async def run_llama_cpp_benchmark():
    """Run comprehensive performance test for llama-cpp-python"""
    
    # Test prompts
    test_input = """<|user|>
You are a fast-food order taker. Your job is to update the user's order based on their request.

Instructions:
- Always start with a polite human-sounding response.
- Only add/remove/replace items if the user clearly asks.
- Only use items from the menu.
- If the user asks for something off-menu, apologize but do not add it.
- If they want more of an item, increase its count.
- Keep existing items unless removed or replaced.
- Format counts as "- 2x Crunchwrap Supreme".
- Do not include any explanation or suggestions.
- Always reflect the updated order accurately — if you say you're adding something, it must appear in the list.
- If a user asks for an item that could be multiple menu items, ask for clarification.

Menu:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell  
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce  
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese  
Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream
Cheese Quesadilla: Flour tortilla, three-cheese blend, creamy jalapeño sauce
Pink Lemonade: Lemonade with red dye
Bottled Water: Purified water

Previous Order:
- 1x Bean Burrito

User said: actually swap the bean burrito for a cheesy gordita crunch and add 2 more tacos throw in a drink too maybe that frozen thing also can I get a quesadilla

<|end|>
<|assistant|>"""

    print("🚀 llama-cpp-python Performance Benchmark")
    print("=" * 60)
    
    # Initialize reasoner
    print("\n🦙 Initializing llama-cpp-python...")
    reasoner = LLMReasoner()
    
    # Multiple test runs for consistency
    runs = 3
    total_times = []
    first_token_times = []
    
    for run in range(1, runs + 1):
        print(f"\n📊 Run {run}/{runs}")
        print("-" * 30)
        
        response = ""
        run_start = time.time()
        first_token_captured = False
        first_token_time = None
        
        async for token in reasoner.generate_response_stream(test_input):
            if not first_token_captured:
                first_token_time = time.time() - run_start
                first_token_captured = True
            print(token, end='', flush=True)
            response += token
        
        total_time = time.time() - run_start
        total_times.append(total_time)
        if first_token_time:
            first_token_times.append(first_token_time * 1000)  # Convert to ms
        
        print(f"\n✅ Run {run} complete: {total_time:.2f}s")
    
    # Calculate averages
    avg_total = sum(total_times) / len(total_times)
    avg_first_token = sum(first_token_times) / len(first_token_times) if first_token_times else 0
    
    # Memory stats
    memory_stats = reasoner.get_memory_stats()
    
    print(f"\n\n📊 Performance Summary ({runs} runs)")
    print("=" * 60)
    print(f"🚀 Average total time: {avg_total:.2f}s")
    print(f"⚡ Average time to first token: {avg_first_token:.0f}ms")
    print(f"🔥 Estimated tokens/sec: ~{len(response.split()) * 2 / avg_total:.1f} tokens/sec")
    print(f"💾 GPU memory usage: {memory_stats.get('gpu_allocated_mb', 0):.1f}MB")
    print(f"🎯 Context utilization: {memory_stats.get('kv_tokens_used', 0)}/{memory_stats.get('kv_tokens_max', 2048)} tokens")
    
    print(f"\n🏆 Performance Rating:")
    if avg_first_token < 50:
        print("⚡ EXCELLENT: Sub-50ms first token!")
    elif avg_first_token < 100:
        print("🚀 VERY GOOD: Sub-100ms first token")
    elif avg_first_token < 200:
        print("✅ GOOD: Sub-200ms first token")
    else:
        print("⏰ SLOW: >200ms first token")
    
    if avg_total < 0.5:
        print("🔥 EXCELLENT: Sub-500ms total response!")
    elif avg_total < 1.0:
        print("🚀 VERY GOOD: Sub-1s total response")
    elif avg_total < 2.0:
        print("✅ GOOD: Sub-2s total response")
    else:
        print("⏰ SLOW: >2s total response")

if __name__ == "__main__":
    asyncio.run(run_llama_cpp_benchmark())
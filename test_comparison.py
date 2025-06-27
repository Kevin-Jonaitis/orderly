#!/usr/bin/env python3
"""
Performance comparison between llama-cpp-python and ExLlamaV2
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.processors.llm import LLMReasoner
from backend.processors.exllama_processor import ExLlamaV2Reasoner

async def run_performance_test():
    """Run performance comparison between both backends"""
    
    # Test prompt
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

    print("🚀 LLM Performance Comparison Test")
    print("=" * 60)
    
    # Test 1: llama-cpp-python
    print("\n🦙 Testing llama-cpp-python (GGUF)")
    print("-" * 40)
    
    try:
        reasoner_cpp = LLMReasoner()
        
        print("\n🌊 llama-cpp streaming test:")
        cpp_response = ""
        cpp_start = time.time()
        async for token in reasoner_cpp.generate_response_stream(test_input):
            print(token, end='', flush=True)
            cpp_response += token
        cpp_total_time = time.time() - cpp_start
        
        print(f"\n✅ llama-cpp complete in {cpp_total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ llama-cpp error: {e}")
        cpp_total_time = None
        cpp_response = ""
    
    # Test 2: ExLlamaV2
    print(f"\n\n🔥 Testing ExLlamaV2 (EXL2)")
    print("-" * 40)
    
    try:
        reasoner_exl = ExLlamaV2Reasoner()
        
        print("\n🌊 ExLlamaV2 streaming test:")
        exl_response = ""
        exl_start = time.time()
        async for token in reasoner_exl.generate_response_stream(test_input):
            print(token, end='', flush=True)
            exl_response += token
        exl_total_time = time.time() - exl_start
        
        print(f"\n✅ ExLlamaV2 complete in {exl_total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ ExLlamaV2 error: {e}")
        exl_total_time = None
        exl_response = ""
    
    # Comparison summary
    print(f"\n\n📊 Performance Summary")
    print("=" * 60)
    
    if cpp_total_time and exl_total_time:
        speedup = cpp_total_time / exl_total_time if exl_total_time > 0 else 0
        winner = "ExLlamaV2" if exl_total_time < cpp_total_time else "llama-cpp"
        
        print(f"🦙 llama-cpp-python: {cpp_total_time:.2f}s")
        print(f"🔥 ExLlamaV2:         {exl_total_time:.2f}s")
        print(f"🏆 Winner: {winner}")
        print(f"⚡ Speedup: {speedup:.2f}x")
        
        # Memory comparison
        if hasattr(reasoner_cpp, 'get_memory_stats') and hasattr(reasoner_exl, 'get_memory_stats'):
            cpp_stats = reasoner_cpp.get_memory_stats()
            exl_stats = reasoner_exl.get_memory_stats()
            
            print(f"\n💾 Memory Usage:")
            print(f"🦙 llama-cpp GPU: {cpp_stats.get('gpu_allocated_mb', 0):.1f}MB")
            print(f"🔥 ExLlamaV2 GPU: {exl_stats.get('gpu_allocated_mb', 0):.1f}MB")
        
        print(f"\n📝 Response Quality:")
        print(f"🦙 llama-cpp: {len(cpp_response)} chars")
        print(f"🔥 ExLlamaV2: {len(exl_response)} chars")
        
    else:
        print("❌ Could not complete comparison due to errors")

if __name__ == "__main__":
    asyncio.run(run_performance_test())
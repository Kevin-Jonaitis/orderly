#!/usr/bin/env python3
"""
Test the eval() + generate() approach vs other methods
"""

import time
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import our LLM processor
from processors.llm import LLMReasoner

async def test_eval_approach():
    """Test the new eval() + generate() approach"""
    
    print("🧪 Testing eval() + generate() Approach")
    print("="*50)
    
    # Initialize LLM processor with eval approach
    print("🔧 Initializing LLM with eval() approach...")
    llm_reasoner = LLMReasoner()
    
    # Test inputs
    test_inputs = [
        "can i get a cheeseburger with fries",
        "actually make that a large fries", 
        "and add a coke please",
        "wait skip the coke, i want lemonade",
        "how much is my total?"
    ]
    
    print(f"\n📊 Running {len(test_inputs)} test requests...")
    
    times = []
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n{'='*30}")
        print(f"Request {i+1}: '{user_input}'")
        print(f"{'='*30}")
        
        start_time = time.time()
        response = await llm_reasoner.generate_response(user_input)
        elapsed = time.time() - start_time
        
        times.append(elapsed)
        
        print(f"📝 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"⏱️  Total time: {elapsed*1000:.0f}ms")
    
    # Analysis
    avg_time = sum(times) / len(times)
    
    print(f"\n{'='*50}")
    print("📊 PERFORMANCE RESULTS")
    print(f"{'='*50}")
    
    print(f"Individual times:")
    for i, (user_input, elapsed) in enumerate(zip(test_inputs, times)):
        expected_speedup = "🚀" if i > 0 else "📋"  # Expect speedup after first request
        print(f"  {i+1}. {user_input[:25]:<25}: {elapsed*1000:>6.0f}ms {expected_speedup}")
    
    print(f"\nAverage time: {avg_time*1000:.0f}ms")
    
    # Compare with expected baseline
    baseline_time = 466  # From our Q4_K_S benchmark
    improvement = (baseline_time - avg_time*1000) / baseline_time * 100
    
    print(f"Previous baseline: {baseline_time}ms")
    print(f"Improvement: {improvement:+.1f}%")
    
    if avg_time * 1000 < 200:
        print(f"🎉 Sub-200ms target achieved! ({avg_time*1000:.0f}ms)")
    elif improvement > 0:
        print(f"✅ Performance improved by {improvement:.1f}%")
    else:
        print(f"❌ Performance regression: {abs(improvement):.1f}% slower")
    
    # Check first vs subsequent requests
    if len(times) > 1:
        first_request = times[0] * 1000
        avg_subsequent = sum(times[1:]) / len(times[1:]) * 1000
        subsequent_improvement = (first_request - avg_subsequent) / first_request * 100
        
        print(f"\n🔍 System Prompt Reuse Analysis:")
        print(f"First request: {first_request:.0f}ms (includes system prompt processing)")
        print(f"Subsequent avg: {avg_subsequent:.0f}ms (system prompt reused)")
        print(f"Reuse benefit: {subsequent_improvement:.1f}% faster")
        
        if subsequent_improvement > 20:
            print(f"✅ Strong evidence of system prompt reuse!")
        elif subsequent_improvement > 0:
            print(f"⚠️  Some system prompt reuse, but less than expected")
        else:
            print(f"❌ No clear system prompt reuse benefit")

if __name__ == "__main__":
    asyncio.run(test_eval_approach())
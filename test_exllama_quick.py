#!/usr/bin/env python3
"""
Quick ExLlamaV2 test with our processor
"""

import asyncio
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.processors.exllama_processor import ExLlamaV2Reasoner

async def test_quick():
    print("🔥 Quick ExLlamaV2 Test")
    print("=" * 30)
    
    try:
        # Initialize reasoner
        print("Initializing ExLlamaV2...")
        start_init = time.time()
        reasoner = ExLlamaV2Reasoner()
        init_time = time.time() - start_init
        print(f"✅ Initialized in {init_time:.2f}s")
        
        # Test with simple prompt
        simple_prompt = "Hello, how are you today?"
        print(f"\nTesting with: '{simple_prompt}'")
        
        start_test = time.time()
        response = ""
        
        async for token in reasoner.generate_response_stream(simple_prompt):
            response += token
            print(token, end='', flush=True)
            
            # Check if we're taking too long
            elapsed = time.time() - start_test
            if elapsed > 5.0:
                print(f"\n⚠️  TIMEOUT after {elapsed:.1f}s")
                break
        
        total_time = time.time() - start_test
        print(f"\n\n📊 Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Response length: {len(response)} chars")
        
        if total_time > 5.0:
            print("❌ FAILED: Too slow")
        else:
            print("✅ SUCCESS: Fast enough")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_quick())
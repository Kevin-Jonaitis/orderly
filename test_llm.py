#!/usr/bin/env python3
"""
Test script for Phi-3 Mini LLM integration
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.processors.llm import LLMReasoner

async def test_llm():
    """Test the LLM with hardcoded input"""
    print("🧠 Testing Phi-3 Mini LLM...")
    
    # Initialize LLM
    print("Loading Phi-3 Mini model...")
    reasoner = LLMReasoner()
    print("✅ Model loaded successfully!")
    

    user_input_one = """
Previous Order:
- 1x Bean Burrito

User said: actually swap the bean burrito for a cheesy gordita crunch and add 2 more tacos throw in a drink too maybe that frozen thing also can I get a quesadilla and a hot sauce oh wait, not hot sauce, the really spicy one. also add a side of dip.

<|end|>
<|assistant|>"""

    user_input_two = """
Previous Order:
- 1x Taco Supreme
- 1x Bean Burrito
- 1x Cheese Quesadilla

User said: okay actually can I have a Cheesy Gordita Crunch instead of the burrito. And can you add a crunchywrap, I love those things. And can you add a chicken salad?

<|end|>
<|assistant|>"""

    # Test streaming inference with real-time output
    print("\n🌊 Testing streaming inference - First input:")
    print("=" * 60)
    print("\n🤖 LLM Response (streaming): ", end='', flush=True)
    
    streamed_response_one = ""
    for token in reasoner.generate_response_stream(user_input_one, None):
        print(token, end='', flush=True)  # Print each token as it arrives
        streamed_response_one += token
    
    print(f"\n\n✅ Complete response: '{streamed_response_one}'")
    
    # Test second input with streaming
    print("\n\n🌊 Testing streaming inference - Second input:")
    print("=" * 60)
    print("\n🤖 LLM Response (streaming): ", end='', flush=True)
    
    streamed_response_two = ""
    for token in reasoner.generate_response_stream(user_input_two, None):
        print(token, end='', flush=True)  # Print each token as it arrives
        streamed_response_two += token
    
    print(f"\n\n✅ Complete response: '{streamed_response_two}'")


    streamed_response_three = ""
    for token in reasoner.generate_response_stream("Can I get a beefy burrito", None):
        print(token, end='', flush=True)  # Print each token as it arrives
        streamed_response_two += token
    
    print(f"\n\n✅ Complete response: '{streamed_response_three}'")

if __name__ == "__main__":
    asyncio.run(test_llm())
    
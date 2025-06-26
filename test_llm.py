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
    
    # Test with hardcoded input
    test_input = """
<|system|>
You're a fast-food order taker. Keep existing items unless cancelled. Be polite and concise, and end your response with:

Final Order:
- Item
- Item
<|end|>
<|user|>
Order so far:
- Waffle

Menu: Cheeseburger, Chicken Sandwich, Fries, Waffle, Coke, Lemonade

can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade
<|end|>
<|assistant|>
"""    
    # Test fast LLM inference
    print("Testing fast LLM inference...")
    response = await reasoner.generate_response(test_input)
    
    print(f"\n🤖 LLM Response: '{response}'")

if __name__ == "__main__":
    asyncio.run(test_llm())
    
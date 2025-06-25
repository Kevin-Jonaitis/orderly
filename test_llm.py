#!/usr/bin/env python3
"""
Test script for Phi-3 Mini LLM integration
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from processors.llm import LLMReasoner

async def test_llm():
    """Test the LLM with hardcoded input"""
    print("🧠 Testing Phi-3 Mini LLM...")
    
    # Initialize LLM
    print("Loading Phi-3 Mini model...")
    reasoner = LLMReasoner()
    print("✅ Model loaded successfully!")
    
    # Test with hardcoded input
    test_input = "I'll have a cheeseburger and fries please"
    print(f"\n🗣  Test input: '{test_input}'")
    
    # Process order
    print("Processing order...")
    order_items = await reasoner.process_order(test_input)
    
    # Generate response
    print("Generating response...")
    response = await reasoner.generate_response(order_items)
    
    print(f"\n✅ Order processed:")
    for item in order_items:
        print(f"  - {item.name}: ${item.price}")
    
    print(f"\n🤖 LLM Response: '{response}'")

if __name__ == "__main__":
    asyncio.run(test_llm())
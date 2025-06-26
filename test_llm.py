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
You are a fast-food order taker. Follow these rules exactly:

- Only accept items from the menu.
- Never add items not listed.
- Never give free substitutions or suggestions.
- If asked for an unavailable item, reply: "Sorry, we don’t have that."
- Always update the list based on the user’s request.
- End your reply with a polite response and with:

Current Order:
- Item
- Item



Menu:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese
Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream
<|end|>
"""

    user_input_one = """<|user|>
Current order:
- Bean Burrito

can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade.
<|end|>
<|assistant|>"""

    user_input_two = """<|user|>
Current order:
- Bean Burrito

okay actually can I have a Cheesy Gordita Crunch instead of the burrito
<|end|>
<|assistant|>"""

    # Test fast LLM inference
    print("Testing fast LLM inference...")
    response = await reasoner.generate_response(test_input + user_input_one)
    
    print(f"\n🤖 LLM Response: '{response}'")

    response_two = await reasoner.generate_response(test_input + user_input_two)
    
    print(f"\n🤖 LLM Response: '{response_two}'")

if __name__ == "__main__":
    asyncio.run(test_llm())
    
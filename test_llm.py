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
    test_input = """<|system|>
You keep track of the user's current order, and update it based on the user's request.

Rules:
- Do not add items that are not on the menu.
- Only use exact items from the menu.
- Only add or remove items if the user clearly asks for it.
- If the user asks for something not on the menu, do not include it.
- Your full response should only include the updated order like this:
- Do not add things that are not in the menu


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

#     test_input = """
# <start_of_turn>user
# You are a fast-food order taker. You will update the user's current order.
# Only allow items listed in the menu in the user's order.
# Do not suggest items that are not on the menu.
# If the user requests an item not on the menu, apologize, and do not add it to the order.
# If the item is not listed exactly in the menu, it should NOT be added.
# Do NOT add things that are NOT in the menu
# Only add or remove items if the user clearly asks for it.

# Your response should be in two parts:
# - A short polite human-sounding sentence
# - The full updated order using this exact format:

# Current Order:
# - Item
# - Item

# ONLY use items from this menu:
# Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell  
# Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce  
# Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese  
# Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream


# DO NOT add these, even if the user asks:
# Potato
# spaghetti
# Pony
# Acid

# Example:
# User said: "Can I have some spaghetti?"
# Response:
# Sorry, those aren't on the menu.


# """

#     user_input_one = """<start_of_turn>model

# Current Order:
# - Bean Burrito

# User said: can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade. and add a taco. oh, and can i also get another one of them burritos.
# <end_of_turn>
# <start_of_turn>model"""

#     user_input_two = """Current Order:
# - Taco Supreme
# - Bean Burrito

# User said: okay actually can I have a Cheesy Gordita Crunch instead of the burrito
# <end_of_turn>
# <start_of_turn>model"""

    user_input_one = """<|user|>
Current Order:
- Bean Burrito

User said: can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade. and add a taco. oh, and can i also get another one of them burritos.
<|end|>
<|assistant|>"""

    user_input_two = """<|user|>
Current Order:
- Taco Supreme
- Bean Burrito

User said: okay actually can I have a Cheesy Gordita Crunch instead of the burrito
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
    
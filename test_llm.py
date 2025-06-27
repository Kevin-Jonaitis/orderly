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


<|user|>
Previous Order:
- 1x Bean Burrito

User said: can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade. and add 2 tacos. oh, and can i also get one more burrito. and a crunchwrap supreme. make that 2 crunchwraps. and can I get a side of sauce?
<|end|>
<|assistant|>
Sorry, we don’t serve cheeseburgers or fries. We’ve added two tacos, another bean burrito, and two Crunchwrap Supremes, and a Pink Lemonade. For sauces, we have several options. Would you like orange sauce, green sauce, or pink sauce?

Updated Order:
- 2x Bean Burrito
- 2x Taco Supreme
- 2x Crunchwrap Supreme
- 1x Pink Lemonade
<|end|>

<|user|>
Previous Order:
- 1x Taco Supreme

User said: can I also get  a bean burrito, and some and a water
<|end|>
<|assistant|>
Absolutely! I added your taco and water. Is there anything else you'd like?

Updated Order:
- 1x Taco Supreme
- 1x Bean Burrito
- 1x Bottled Water
<|end|>
<|user|>


Menu:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell  
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce  
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese  
Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream
Cheese Quesadilla: Flour tortilla, three-cheese blend, creamy jalapeño sauce
Pink Lemonade: Lemonade with red dye
Tropicana Orange Juice: 100% orange juice
Bottled Water: Purified water
G2 Gatorade Fruit Punch: Electrolyte drink
Frozen Baja Blast: Frozen lime soda slush
Strawberry Skittles Freeze: Frozen drink with Skittles flavor
Nacho Cheese Dip: Melted cheese
Guacamole Dip: Mashed avocado with spices
Pico de Gallo: Chopped tomatoes, onions, cilantro, lime juice
Avocado Ranch Sauce: Creamy ranch with avocado flavor
Creamy Jalapeño Sauce: Spicy, creamy jalapeño blend
Red Sauce: Mild enchilada-style sauce
Fire Sauce Packet: Very spicy sauce
Hot Sauce Packet: Spicy sauce
Mild Sauce Packet: Mildly spicy sauce
Diablo Sauce Packet: Extra spicy sauce
Grilled Chicken Taco: Grilled chicken, lettuce, cheddar cheese, soft tortilla
Double Decker Taco: Crunchy taco with refried beans and soft tortilla
Loaded Nacho Taco: Seasoned beef, nacho cheese, lettuce, red tortilla strips, soft tortilla
Spicy Potato Soft Taco: Seasoned potatoes, lettuce, cheddar cheese, chipotle sauce, soft tortilla
Triple Layer Nachos: Chips, refried beans, red sauce, nacho cheese
Beefy 5-Layer Burrito: Ground beef, nacho cheese, cheddar cheese, refried beans, sour cream, flour tortilla
XXL Grilled Stuft Burrito: Ground beef, rice, beans, guacamole, pico de gallo, cheddar cheese, sour cream

Now update the order based on the user request below.
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

    # Test fast LLM inference
    print("Testing fast LLM inference...")
    response = await reasoner.generate_response(test_input + user_input_one)
    
    print(f"\n🤖 LLM Response: '{response}'")

    response_two = await reasoner.generate_response(test_input + user_input_two)
    
    print(f"\n🤖 LLM Response: '{response_two}'")

if __name__ == "__main__":
    asyncio.run(test_llm())
    
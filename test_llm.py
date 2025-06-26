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
You're a fast-food order taker.

Menu:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese
Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream
Quesarito: Flour tortilla, seasoned beef, cheddar cheese, nacho cheese, sour cream, rice
Soft Taco: Flour tortilla, seasoned beef, lettuce, cheddar cheese
Hard Taco: Crunchy corn shell, seasoned beef, lettuce, cheddar cheese
Chicken Soft Taco: Flour tortilla, grilled chicken, lettuce, cheddar cheese
Shredded Chicken Burrito: Flour tortilla, shredded chicken, rice, cheddar cheese, avocado ranch
Steak Quesadilla: Grilled steak, cheddar cheese, three-cheese blend, creamy jalapeño sauce, flour tortilla
Cheese Quesadilla: Flour tortilla, three-cheese blend, creamy jalapeño sauce
Black Bean Quesarito: Black beans, rice, nacho cheese, sour cream, cheddar cheese, flour tortilla
Power Bowl (Chicken): Grilled chicken, black beans, rice, guacamole, lettuce, avocado ranch, pico de gallo, sour cream, cheddar cheese
Power Bowl (Veggie): Black beans, rice, guacamole, lettuce, avocado ranch, pico de gallo, sour cream, cheddar cheese
Nachos BellGrande: Tortilla chips, seasoned beef, refried beans, nacho cheese, sour cream, diced tomatoes
Cheesy Fiesta Potatoes: Seasoned potato bites, nacho cheese, sour cream
Mexican Pizza: Two tortillas, seasoned beef, refried beans, pizza sauce, three-cheese blend, diced tomatoes
Spicy Tostada: Flat tostada shell, refried beans, red sauce, lettuce, tomato, cheddar cheese
Cinnamon Twists: Fried pastry, cinnamon sugar
Cinnabon Delights (2 Pack): Fried dough balls, cream cheese filling, cinnamon sugar
Chalupa Supreme (Beef): Chalupa shell, ground beef, lettuce, tomatoes, sour cream, cheddar cheese
Chalupa Supreme (Chicken): Chalupa shell, grilled chicken, lettuce, tomatoes, sour cream, cheddar cheese
Chalupa Supreme (Veggie): Chalupa shell, black beans, lettuce, tomatoes, sour cream, cheddar cheese
Doritos Locos Taco (Nacho Cheese): Doritos taco shell, seasoned beef, lettuce, cheddar cheese
Doritos Locos Taco Supreme: Doritos taco shell, seasoned beef, lettuce, tomato, sour cream, cheddar cheese
Burrito Supreme (Beef): Flour tortilla, ground beef, refried beans, sour cream, tomato, lettuce, red sauce, cheddar cheese, onions
Burrito Supreme (Steak): Flour tortilla, grilled steak, refried beans, sour cream, tomato, lettuce, red sauce, cheddar cheese, onions
Black Bean Crunchwrap Supreme: Flour tortilla, black beans, tostada shell, nacho cheese, lettuce, tomato, sour cream
Mini Skillet Bowl (Breakfast): Scrambled eggs, nacho cheese, potato bites, pico de gallo
Grilled Breakfast Burrito (Sausage): Flour tortilla, scrambled eggs, sausage crumbles, nacho cheese
Grilled Breakfast Burrito (Bacon): Flour tortilla, scrambled eggs, bacon, nacho cheese
Hash Browns: Fried potato patty
Breakfast Crunchwrap (Bacon): Flour tortilla, scrambled eggs, hash brown, bacon, cheddar cheese, creamy jalapeño
Breakfast Crunchwrap (Sausage): Flour tortilla, scrambled eggs, hash brown, sausage, cheddar cheese, creamy jalapeño
Mountain Dew Baja Blast: Tropical lime soda
Pepsi: Cola
Diet Pepsi: Diet cola
Dr Pepper: Spiced cola soda
Wild Cherry Pepsi: Cherry-flavored cola
Lipton Brisk Iced Tea: Sweetened black tea
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
XXL Grilled Stuft Burrito: Ground beef, rice, beans, guacamole, pico de gallo, cheddar cheese, sour cream, flour tortilla
Enchirito (Discontinued): Tortilla, ground beef, beans, red sauce, melted cheese
Volcano Taco (Discontinued): Spicy taco shell, seasoned beef, lava sauce, lettuce, cheese
Nacho Fries (Seasonal): Seasoned fries with nacho cheese dip
Nacho Fries Supreme: Nacho fries, seasoned beef, nacho cheese, sour cream, diced tomatoes
Mexican Rice: Seasoned rice side
Refried Beans: Pinto beans mashed and seasoned
Churros (Discontinued): Fried dough sticks with cinnamon sugar
Caramel Apple Empanada (Discontinued): Pastry filled with apple and caramel
Fiesta Taco Salad: Crispy tortilla bowl, lettuce, tomato, beef, beans, red tortilla strips, sour cream, cheddar cheese
Chicken Enchilada Bowl (Discontinued): Grilled chicken, rice, enchilada sauce, cheese, sour cream
Border Bowl (Veggie): Black beans, rice, guacamole, lettuce, tomato, cheese
Meximelt (Discontinued): Soft tortilla, beef, pico de gallo, melted cheese
Beefy Melt Burrito: Seasoned beef, three-cheese blend, rice, sour cream, tortilla strips
Chicken Chipotle Melt: Grilled chicken, chipotle sauce, shredded cheese, flour tortilla
Loaded Chicken Flatbread Taco: Flatbread, grilled chicken, cheddar cheese, chipotle sauce
Beefy Potato Flatbread Taco: Flatbread, ground beef, seasoned potatoes, nacho cheese
Cheesy Roll-Up: Flour tortilla, melted three-cheese blend
Veggie Burrito: Rice, black beans, lettuce, tomato, sour cream, cheddar cheese, flour tortilla
Grilled Cheese Burrito: Seasoned beef, rice, nacho cheese, chipotle sauce, three-cheese blend, tortilla grilled with cheese
Fiery Doritos Taco: Spicy Doritos shell, beef, lettuce, cheese
Fresco Soft Taco (Discontinued): Soft taco with pico de gallo instead of cheese or sauce
Lava Sauce (Discontinued): Spicy cheese sauce
Smothered Burrito (Discontinued): Burrito topped with red sauce, melted cheese, and sour cream

Keep existing items unless cancelled. Be polite and concise, and end your response with

Final Order:
- Item
- Item
<|end|>
"""

    user_input_one = """<|user|>
Order so far:
- Waffle


can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade
<|end|>
<|assistant|>"""

    user_input_two = """<|user|>
Order so far:
- Waffle
- Cheeseburger
- Fries
- Lemonade

okay actually can you remove the waffle
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
    
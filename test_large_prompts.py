#!/usr/bin/env python3
"""
Test large prompt handling with different sizes
"""

import time
import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import our LLM processor
from processors.llm import LLMReasoner

async def test_large_prompts():
    """Test prompts of various sizes to verify coherent output"""
    
    print("🧪 Large Prompt Test Suite")
    print("="*50)
    
    # Initialize LLM processor
    print("🔧 Initializing LLM processor...")
    llm_reasoner = LLMReasoner()
    
    # Test cases with different prompt sizes
    test_cases = [
        {
            "name": "Small Prompt (100-200 tokens)",
            "prompt": """You are a helpful AI assistant. Please respond to the following request clearly and concisely.

User request: What is the capital of France?"""
        },
        {
            "name": "Medium Prompt (500-700 tokens)", 
            "prompt": """You are an AI fast-food order taker for a popular restaurant chain. You are friendly, efficient, and knowledgeable about the menu. Your goal is to help customers place their orders accurately and provide excellent customer service.

Here is the current menu with prices:

BURGERS & SANDWICHES:
- Classic Cheeseburger: $8.99
- Bacon Cheeseburger: $10.99  
- Deluxe Burger with lettuce, tomato, onion: $9.99
- Grilled Chicken Sandwich: $8.49
- Fish Sandwich: $7.99

SIDES:
- French Fries (Small): $2.99
- French Fries (Large): $3.99
- Onion Rings: $3.49
- Mozzarella Sticks: $4.99

DRINKS:
- Soft Drinks (Small): $1.99
- Soft Drinks (Large): $2.49
- Coffee: $1.49
- Milkshakes: $3.99

DESSERTS:
- Apple Pie: $2.99
- Chocolate Chip Cookies (3 pack): $2.49

Current customer order: Empty

Customer says: "Hi, I'd like to order a cheeseburger meal with fries and a coke, please."""
        },
        {
            "name": "Large Prompt (1000-1500 tokens)",
            "prompt": """You are an AI fast-food order taker for a premium restaurant chain called "Golden Grill Express." You are friendly, efficient, and extremely knowledgeable about the extensive menu offerings. Your goal is to help customers place their orders accurately, suggest complementary items, and provide excellent customer service while maintaining efficiency during busy periods.

RESTAURANT BACKGROUND:
Golden Grill Express is known for fresh ingredients, customizable options, and quick service. We pride ourselves on accuracy and customer satisfaction. All items are made to order with fresh, locally-sourced ingredients when possible.

COMPLETE MENU WITH DETAILED DESCRIPTIONS AND PRICES:

SIGNATURE BURGERS & SANDWICHES:
- Classic Golden Cheeseburger: $8.99 (Angus beef patty, American cheese, lettuce, tomato, pickles, special sauce)
- Premium Bacon Cheeseburger: $10.99 (Double Angus patty, thick-cut bacon, cheddar cheese, lettuce, tomato, onion)
- Deluxe Garden Burger: $9.99 (Angus beef, Swiss cheese, lettuce, tomato, red onion, avocado, mayo)
- Spicy Jalapeño Burger: $9.49 (Angus beef, pepper jack cheese, jalapeños, lettuce, chipotle mayo)
- Grilled Chicken Deluxe: $8.49 (Marinated chicken breast, Swiss cheese, lettuce, tomato, honey mustard)
- Crispy Chicken Sandwich: $7.99 (Hand-breaded chicken, pickles, mayo, brioche bun)
- Fish & Chips Sandwich: $7.99 (Beer-battered cod, tartar sauce, lettuce, sesame bun)

SIDES & APPETIZERS:
- Golden Fries (Small): $2.99 (Hand-cut, seasoned)
- Golden Fries (Large): $3.99 (Hand-cut, seasoned)
- Sweet Potato Fries: $3.49 (With cinnamon sugar)
- Onion Rings: $3.49 (Beer-battered, thick-cut)
- Mozzarella Sticks (6 pieces): $4.99 (With marinara sauce)
- Chicken Wings (8 pieces): $6.99 (Choice of sauce: Buffalo, BBQ, Honey Garlic)
- Loaded Nachos: $5.99 (Cheese, jalapeños, sour cream, guacamole)

BEVERAGES:
- Soft Drinks (Small): $1.99 (Coke, Pepsi, Sprite, Orange, Root Beer)
- Soft Drinks (Large): $2.49 (Coke, Pepsi, Sprite, Orange, Root Beer)
- Fresh Coffee: $1.49 (Regular or Decaf)
- Premium Milkshakes: $3.99 (Vanilla, Chocolate, Strawberry, Oreo)
- Fresh Lemonade: $2.29
- Iced Tea: $1.99

DESSERTS:
- Homemade Apple Pie: $2.99 (Served warm with cinnamon)
- Chocolate Chip Cookies (3 pack): $2.49 (Fresh baked daily)
- Ice Cream Sundae: $3.49 (Vanilla ice cream, chocolate sauce, whipped cream, cherry)

CURRENT CUSTOMER ORDER STATUS: Empty (new customer)

Customer walks up to the counter and says: "Hi there! I'm really hungry and looking for a good meal. What would you recommend? I like burgers but I'm also interested in trying something new. And I'd definitely like some fries and a drink with that."""
        },
        {
            "name": "Very Large Prompt (2000+ tokens)",
            "prompt": """You are an AI fast-food order taker for "Golden Grill Express," an upscale quick-service restaurant chain known for premium ingredients, exceptional customer service, and innovative menu offerings. You have been specifically trained to handle complex orders, dietary restrictions, and provide detailed recommendations based on customer preferences.

RESTAURANT MISSION & VALUES:
Golden Grill Express is committed to serving fresh, high-quality food made with locally-sourced ingredients whenever possible. We pride ourselves on customization options, accommodating dietary restrictions, and maintaining the highest standards of food safety and customer service. Our team is trained to be knowledgeable, friendly, and efficient while ensuring every customer has an exceptional dining experience.

COMPREHENSIVE MENU WITH DETAILED NUTRITIONAL INFORMATION:

SIGNATURE BURGERS & GOURMET SANDWICHES:
- Classic Golden Cheeseburger: $8.99 
  (Premium Angus beef patty 1/4 lb, aged American cheese, crisp lettuce, fresh tomato, house pickles, Golden special sauce, toasted brioche bun)
  [Calories: 650, Protein: 35g, Carbs: 45g, Fat: 35g]

- Premium Bacon Cheeseburger: $10.99 
  (Double Angus beef patty 1/2 lb total, thick-cut applewood bacon, aged cheddar cheese, lettuce, tomato, red onion, brioche bun)
  [Calories: 920, Protein: 55g, Carbs: 48g, Fat: 52g]

- Deluxe Garden Burger: $9.99 
  (Angus beef patty 1/4 lb, Swiss cheese, organic lettuce, heirloom tomato, red onion, fresh avocado, garlic aioli, whole grain bun)
  [Calories: 720, Protein: 38g, Carbs: 52g, Fat: 38g]

- Spicy Jalapeño Burger: $9.49 
  (Angus beef patty 1/4 lb, pepper jack cheese, fresh jalapeños, lettuce, chipotle mayo, toasted brioche bun)
  [Calories: 680, Protein: 36g, Carbs: 46g, Fat: 36g]

- Grilled Chicken Deluxe: $8.49 
  (Marinated chicken breast 6oz, Swiss cheese, lettuce, tomato, honey mustard, whole grain bun)
  [Calories: 520, Protein: 42g, Carbs: 38g, Fat: 22g]

- Crispy Chicken Sandwich: $7.99 
  (Hand-breaded chicken breast, house pickles, mayo, brioche bun)
  [Calories: 590, Protein: 32g, Carbs: 48g, Fat: 28g]

- Beer-Battered Fish Sandwich: $7.99 
  (Pacific cod, tartar sauce, lettuce, sesame bun)
  [Calories: 480, Protein: 28g, Carbs: 42g, Fat: 24g]

VEGETARIAN & VEGAN OPTIONS:
- Beyond Burger: $9.99 (Plant-based patty, vegan cheese, lettuce, tomato, vegan mayo, whole grain bun)
- Portobello Mushroom Sandwich: $8.49 (Grilled portobello, Swiss cheese, roasted peppers, balsamic glaze)

SIDES & APPETIZERS:
- Golden Hand-Cut Fries (Small): $2.99 [Calories: 320]
- Golden Hand-Cut Fries (Large): $3.99 [Calories: 480]
- Sweet Potato Fries: $3.49 (With cinnamon sugar) [Calories: 340]
- Beer-Battered Onion Rings: $3.49 [Calories: 380]
- Mozzarella Sticks (6 pieces): $4.99 (With marinara sauce) [Calories: 520]
- Buffalo Chicken Wings (8 pieces): $6.99 (Choice of sauce: Buffalo, BBQ, Honey Garlic, Teriyaki) [Calories: 480-520]
- Loaded Nachos Supreme: $5.99 (Cheese sauce, jalapeños, sour cream, guacamole, salsa) [Calories: 680]
- Garden Salad: $4.99 (Mixed greens, tomato, cucumber, carrots, choice of dressing) [Calories: 120-220]

BEVERAGES:
- Soft Drinks (Small/Large): $1.99/$2.49 (Coke, Pepsi, Sprite, Orange, Root Beer, Diet options available)
- Fresh Ground Coffee: $1.49 (Regular, Decaf, French Roast)
- Premium Milkshakes: $3.99 (Vanilla, Chocolate, Strawberry, Oreo, Peanut Butter Cup) [Calories: 520-650]
- Fresh Squeezed Lemonade: $2.29 [Calories: 150]
- Unsweetened Iced Tea: $1.99 [Calories: 5]
- Fresh Fruit Smoothies: $3.49 (Strawberry Banana, Mango Passion, Berry Blast) [Calories: 280-320]

DESSERTS:
- Homemade Apple Pie: $2.99 (Served warm with cinnamon, optional vanilla ice cream +$1) [Calories: 340]
- Fresh-Baked Chocolate Chip Cookies (3 pack): $2.49 [Calories: 450 total]
- Premium Ice Cream Sundae: $3.49 (Vanilla ice cream, choice of chocolate or caramel sauce, whipped cream, cherry) [Calories: 420]

CURRENT CUSTOMER INTERACTION:
You are serving a customer who appears to be a first-time visitor to Golden Grill Express. They seem interested in learning about the restaurant and are looking for recommendations. They mentioned they have about 20 minutes for lunch and want something satisfying but not too heavy.

The customer approaches your counter and says: "Hi! This is my first time here and I've heard great things about this place. I'm looking for something delicious for lunch - I like burgers and chicken, but I'm also trying to eat a bit healthier these days. I have about 20 minutes, so nothing too complicated. What would you recommend? And I'd definitely like something to drink and maybe a side. Oh, and I should mention I'm not a huge fan of really spicy food, but I do like some flavor. What do you think would be perfect for me?" """
        }
    ]
    
    # Run tests
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            response = await llm_reasoner.generate_response(test_case['prompt'])
            elapsed = time.time() - start_time
            
            # Check if response is coherent (not gibberish)
            is_coherent = check_response_quality(response)
            
            results.append({
                'name': test_case['name'],
                'time_ms': elapsed * 1000,
                'response_length': len(response),
                'coherent': is_coherent,
                'response_preview': response[:200] + "..." if len(response) > 200 else response
            })
            
            print(f"\n📝 Response Preview:")
            print(f"{response[:300]}{'...' if len(response) > 300 else ''}")
            print(f"\n⏱️  Time: {elapsed*1000:.0f}ms")
            print(f"📏 Response length: {len(response)} characters")
            print(f"🧠 Coherent: {'✅ Yes' if is_coherent else '❌ No (likely gibberish)'}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'coherent': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['name']}: ERROR - {result['error']}")
        else:
            status = "✅ PASS" if result['coherent'] else "❌ FAIL (gibberish)"
            print(f"{status} {result['name']}: {result['time_ms']:.0f}ms, {result['response_length']} chars")
    
    # Overall assessment
    passed_tests = sum(1 for r in results if r.get('coherent', False))
    total_tests = len(results)
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All large prompt tests PASSED! No more gibberish!")
    else:
        print("⚠️  Some tests failed - large prompts still producing gibberish")

def check_response_quality(response: str) -> bool:
    """Check if response appears coherent vs gibberish"""
    if not response or len(response) < 10:
        return False
    
    # Check for gibberish patterns
    gibberish_indicators = [
        # Repeated fragments
        response.count('chew') > 3,
        response.count('wheat') > 5,
        response.count('tep') > 3,
        # Fragmented words
        len([word for word in response.split() if len(word) < 3]) > len(response.split()) * 0.3,
        # Excessive repetition
        any(fragment * 3 in response.lower() for fragment in ['te', 'ch', 'we', 'ar']),
        # No proper sentences
        '.' not in response and '?' not in response and '!' not in response
    ]
    
    # If too many gibberish indicators, consider it incoherent
    return sum(gibberish_indicators) < 2

if __name__ == "__main__":
    asyncio.run(test_large_prompts())
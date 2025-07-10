#!/usr/bin/env python3
"""
Test script for fuzzy matching implementation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.order_tracker import OrderTracker

def test_fuzzy_matching():
    """Test the fuzzy matching with various LLM response variations"""
    
    print("🧪 Testing Fuzzy Matching Implementation")
    print("=" * 50)
    
    # Create order tracker
    tracker = OrderTracker()
    
    # Test cases: (LLM response, expected menu item)
    test_cases = [
        ("Chicken Tenders", "Tenders 3pc"),
        ("tenders", "Tenders 3pc"),
        ("3 piece tenders", "Tenders 3pc"),
        ("Medium Pepsi", "Pepsi Medium"),
        ("pepsi medium", "Pepsi Medium"),
        ("Fries", "Fries"),
        ("French Fries", "Fries"),
        ("Chicken Sandwich", "Classic Chicken Sandwich Only"),
        ("Spicy Chicken", "Spicy Chicken Sandwich Only"),
        ("Fried Chicken", "Fried Chicken 2pc"),
        ("2 piece chicken", "Fried Chicken 2pc"),
        ("Mashed Potatoes", "Mashed Potatoes"),
        ("Apple Pie", "Fried Apple Pie"),
        ("Chicken Littles", "Chicken Littles Sandwich Only"),
        ("Box Meal", "5pc Tenders Box Meal"),  # Should match closest box meal
    ]
    
    print(f"📋 Loaded {len(tracker.normalized_menu_items)} normalized menu items")
    print(f"💰 Loaded {len(tracker.menu_prices)} menu prices")
    
    print("\n🔍 Testing fuzzy matches:")
    print("-" * 50)
    
    for llm_response, expected_menu_item in test_cases:
        matched_item, confidence = tracker._find_best_menu_match(llm_response)
        print(f"LLM: '{llm_response}'")
        print(f"  → Matched: '{matched_item}' (confidence: {confidence}%)")
        print(f"  → Expected: '{expected_menu_item}'")
        print(f"  → {'✅' if matched_item == expected_menu_item else '❌'}")
        print()

if __name__ == "__main__":
    test_fuzzy_matching() 
import re
import os
from typing import Dict, List, Any, Tuple
from multiprocessing.managers import BaseManager
from fuzzywuzzy import fuzz

class OrderTracker:
    """Tracks the user's current order by parsing LLM responses"""
    
    def __init__(self):
        self.order_items = {}  # Map from item name to quantity
        self.menu_prices = {}  # Map from item name to price
        self.menu_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'menus', 'menu_items_prices.txt')
        self.menu_descriptions_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'menus', 'menu_items_descriptions.txt')
        
        # In-memory normalized menu for fuzzy matching
        self.normalized_menu_items = {}  # normalized_name -> original_menu_name
        self._load_menu_items()
    
    def _normalize_item_name(self, item_name: str) -> str:
        """Normalize item name for better matching, preserving # for combos"""
        normalized = item_name.lower()
        
        # Remove descriptions (everything after colon)
        if ':' in normalized:
            normalized = normalized.split(':')[0].strip()
        
        # Remove punctuation except #, and extra whitespace
        normalized = re.sub(r'[^\w\s#]', ' ', normalized)
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _load_menu_items(self):
        """Load menu items and create normalized versions for fuzzy matching"""
        self.menu_prices.clear()
        self.normalized_menu_items.clear()
        
        try:
            # Load menu prices
            if os.path.exists(self.menu_file_path):
                with open(self.menu_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if ':' in line and '$' in line:
                            parts = line.split(':')
                            if len(parts) == 2:
                                item_name = parts[0].strip()
                                price_str = parts[1].strip()
                                price_match = re.search(r'\$(\d+\.\d+)', price_str)
                                if price_match:
                                    price = float(price_match.group(1))
                                    self.menu_prices[item_name] = price
                                    
                                    # Create normalized version for fuzzy matching
                                    normalized_name = self._normalize_item_name(item_name)
                                    self.normalized_menu_items[normalized_name] = item_name
                
                print(f"📋 Loaded {len(self.menu_prices)} menu items with prices")
                print(f"🔍 Created {len(self.normalized_menu_items)} normalized menu items for fuzzy matching")
            else:
                print(f"⚠️ Menu file not found: {self.menu_file_path}")
        except Exception as e:
            print(f"❌ Error loading menu items: {e}")
    
    def _find_best_menu_match(self, item_name: str) -> Tuple[str, float]:
        """
        Find the best matching menu item using fuzzy string matching
        
        Returns:
        - (matched_menu_name, confidence_score) - always returns the best match
        """
        normalized_input = self._normalize_item_name(item_name)
        best_match = item_name  # Default to original name
        best_score = 0
        
        print(f" [DEBUG] Normalized input: '{normalized_input}'")
        
        for normalized_menu_name, original_menu_name in self.normalized_menu_items.items():
            # Try multiple fuzzy matching strategies and take the best score
            
            # 1. Token sort ratio (handles word order differences)
            token_sort_score = fuzz.token_sort_ratio(normalized_input, normalized_menu_name)
            
            # 2. Partial ratio (handles partial matches like "chicken" in "chicken sandwich")
            partial_score = fuzz.partial_ratio(normalized_input, normalized_menu_name)
            
            # 3. Token set ratio (handles extra/missing words)
            token_set_score = fuzz.token_set_ratio(normalized_input, normalized_menu_name)
            
            # 4. Simple ratio (exact character matching)
            simple_score = fuzz.ratio(normalized_input, normalized_menu_name)
            
            # Take the highest score from all strategies
            score = max(token_sort_score, partial_score, token_set_score, simple_score)
            
            # Debug logging for combo items
            if "combo" in normalized_menu_name.lower() and score > 50:
                print(f"🔍 [DEBUG] '{normalized_input}' vs '{normalized_menu_name}' = {score}% (token_sort={token_sort_score}, partial={partial_score}, token_set={token_set_score}, simple={simple_score})")
            
            if score > best_score:
                best_score = score
                best_match = original_menu_name
        
        # Log the matching decision
        print(f"🎯 Fuzzy match: '{item_name}' -> '{best_match}' (confidence: {best_score}%)")
        
        return best_match, best_score

    def _clean_item_name(self, item_name: str) -> str:
        """Clean item name by removing description if present"""
        # If the item name contains a colon, it likely has a description
        # Take only the part before the colon
        if ':' in item_name:
            # Split on colon and take the first part
            cleaned_name = item_name.split(':')[0].strip()
            print(f" Cleaned item name: '{item_name}' -> '{cleaned_name}'")
            return cleaned_name
        return item_name

    def parse_and_update_order(self, response_text):
        """Parse response text and update the order items"""
        # Re-read menu prices in case the file has been updated
        self._load_menu_items() # Re-load menu items for fuzzy matching
        
        # Look for "Updated Order:" section
        if "Updated Order:" not in response_text:
            return
        
        # Extract everything after "Updated Order:"
        order_section = response_text.split("Updated Order:")[1]
        
        # Clear current order
        self.order_items.clear()
        
        # Parse each line that starts with "-"
        lines = order_section.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                # Remove the "-" and parse the item
                item_text = line[1:].strip()
                
                # Skip empty items
                if not item_text or item_text == '(empty)':
                    continue
                
                # Parse quantity and item name
                # Pattern: "2x Bean Burrito" -> quantity=2, item="Bean Burrito"
                # Pattern: "Combo #1" -> quantity=1, item="Combo #1"
                match = re.match(r'(\d+)x\s+(.+)', item_text)
                if match:
                    quantity = int(match.group(1))
                    item_name = match.group(2).strip()
                    # Remove any description (anything after :)
                    item_name = item_name.split(':')[0].strip()
                    # Use fuzzy matching to find the best menu item
                    best_match_name, confidence = self._find_best_menu_match(item_name)
                    if best_match_name:
                        self.order_items[best_match_name] = quantity
                    else:
                        self.order_items[item_name] = quantity
                else:
                    item_name = item_text
                    # Remove any description (anything after :)
                    item_name = item_name.split(':')[0].strip()
                    best_match_name, confidence = self._find_best_menu_match(item_name)
                    if best_match_name:
                        self.order_items[best_match_name] = 1
                    else:
                        self.order_items[item_name] = 1
    
    def get_order_summary(self):
        """Return a formatted string of the current order"""
        if not self.order_items:
            return "Previous Order:\n- (empty)"
        
        summary = "Previous Order:\n"
        for item_name, quantity in self.order_items.items():
            summary += f"- {quantity}x {item_name}\n"
        return summary.strip()
    
    def format_order_for_frontend(self) -> Dict[str, Any]:
        """Format order for frontend consumption"""
        if not self.order_items:
            return {"items": [], "total": 0}
        
        items = []
        total = 0
        
        for item_name, quantity in self.order_items.items():
            # Get actual price from menu, fallback to 0.0 if not found
            price = self.menu_prices.get(item_name, 0.0)
            
            # Always strip description from item name before sending to frontend
            clean_name = item_name.split(':')[0].strip()
            
            # Create item object with actual price
            item = {
                "id": f"item_{len(items)}",
                "name": clean_name,
                "price": price,
                "quantity": quantity
            }
            items.append(item)
            total += item["price"] * quantity
        
        return {"items": items, "total": round(total, 2)}
    
    def clear_order(self):
        """Clear the current order"""
        self.order_items.clear()

# Create a custom manager class that registers OrderTracker
class OrderTrackerManager(BaseManager):
    pass

# Register the OrderTracker class with our custom manager
OrderTrackerManager.register('OrderTracker', OrderTracker) 
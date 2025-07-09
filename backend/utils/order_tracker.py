import re
import os
from typing import Dict, List, Any
from multiprocessing.managers import BaseManager

class OrderTracker:
    """Tracks the user's current order by parsing LLM responses"""
    
    def __init__(self):
        self.order_items = {}  # Map from item name to quantity
        self.menu_prices = {}  # Map from item name to price
        self.menu_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'menus', 'menu_items_prices.txt')
    
    def _load_menu_prices(self):
        """Load menu prices from the menu file"""
        self.menu_prices.clear()
        try:
            if os.path.exists(self.menu_file_path):
                with open(self.menu_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if ':' in line and '$' in line:
                            # Parse "Item Name: $X.XX" format
                            parts = line.split(':')
                            if len(parts) == 2:
                                item_name = parts[0].strip()
                                price_str = parts[1].strip()
                                # Extract price from "$X.XX" format
                                price_match = re.search(r'\$(\d+\.\d+)', price_str)
                                if price_match:
                                    price = float(price_match.group(1))
                                    self.menu_prices[item_name] = price
                print(f"📋 Loaded {len(self.menu_prices)} menu items with prices")
            else:
                print(f"⚠️ Menu file not found: {self.menu_file_path}")
        except Exception as e:
            print(f"❌ Error loading menu prices: {e}")
    
    def parse_and_update_order(self, response_text):
        """Parse response text and update the order items"""
        # Re-read menu prices in case the file has been updated
        self._load_menu_prices()
        
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
                    self.order_items[item_name] = quantity
                else:
                    # If no quantity prefix, assume quantity of 1
                    item_name = item_text
                    self.order_items[item_name] = 1
    
    def get_order_summary(self):
        """Return a formatted string of the current order"""
        if not self.order_items:
            return "Previous Order:\n- (empty)"
        
        summary = "Previous Order:\n"
        for item_name, quantity in self.order_items.items():
            price = self.menu_prices.get(item_name, 0.0)
            summary += f"- {quantity}x {item_name} (${price:.2f} each)\n"
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
            
            # Create item object with actual price
            item = {
                "id": f"item_{len(items)}",
                "name": item_name,
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
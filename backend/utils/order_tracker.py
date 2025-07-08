import re
from typing import Dict


class OrderTracker:
    """Tracks the user's current order by parsing LLM responses"""
    
    def __init__(self):
        self.order_items = {}  # Map from item name to quantity
    
    def parse_and_update_order(self, response_text):
        """Parse response text and update the order items"""
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
                match = re.match(r'(\d+)x\s+(.+)', item_text)
                if match:
                    quantity = int(match.group(1))
                    item_name = match.group(2).strip()
                    self.order_items[item_name] = quantity
    
    def get_order_summary(self):
        """Return a formatted string of the current order"""
        if not self.order_items:
            return "Previous Order:\n- (empty)"
        
        summary = "Previous Order:\n"
        for item_name, quantity in self.order_items.items():
            summary += f"- {quantity}x {item_name}\n"
        return summary.strip()
    
    def clear_order(self):
        """Clear the current order"""
        self.order_items.clear() 
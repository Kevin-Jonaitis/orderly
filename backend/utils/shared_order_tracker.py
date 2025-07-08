import re

class SharedOrderTracker:
    def __init__(self, shared_state):
        self.shared_state = shared_state

    def parse_and_update_order(self, response_text):
        if "Updated Order:" not in response_text:
            return
        order_section = response_text.split("Updated Order:")[1]
        self.shared_state.order_items.clear()
        lines = order_section.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                item_text = line[1:].strip()
                if not item_text or item_text == '(empty)':
                    continue
                match = re.match(r'(\d+)x\s+(.+)', item_text)
                if match:
                    quantity = int(match.group(1))
                    item_name = match.group(2).strip()
                    self.shared_state.order_items[item_name] = quantity

    def get_order_summary(self):
        if not self.shared_state.order_items:
            return "Previous Order:\n- (empty)"
        summary = "Previous Order:\n"
        for item_name, quantity in self.shared_state.order_items.items():
            summary += f"- {quantity}x {item_name}\n"
        return summary.strip()

    def clear_order(self):
        self.shared_state.order_items.clear() 
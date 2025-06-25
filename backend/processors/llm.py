"""
Large Language Model (LLM) reasoning processor for order processing.

This module provides:
- LLMReasoner: Handles menu context and order reasoning logic
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OrderItem:
    """Order item data model"""
    
    def __init__(self, id: str, name: str, price: float, quantity: int = 1):
        self.id = id
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "quantity": self.quantity
        }

class LLMReasoner:
    """Stub for Phi-3 Mini reasoning LLM"""
    
    def __init__(self):
        self.menu_context = self.load_menu_context()
    
    def load_menu_context(self) -> str:
        """Load menu context from uploaded files"""
        menu_files = list(Path("menus").glob("*.txt"))
        if not menu_files:
            return "Default menu: Cheeseburger ($8.99), Fries ($3.99), Drink ($2.99)"
        
        context = ""
        for file in menu_files:
            context += file.read_text() + "\n"
        return context
    
    async def process_order(self, text: str) -> List[OrderItem]:
        """Process user text into order items"""
        start_time = time.time()
        
        # Simulate LLM processing
        await asyncio.sleep(0.2)
        
        # Mock order processing
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM_REASONING: {latency_ms:.0f}ms - {text} -> {len(items)} items")
        
        return items
    
    async def generate_response(self, order_items: List[OrderItem]) -> str:
        """Generate response text for TTS"""
        start_time = time.time()
        
        await asyncio.sleep(0.1)
        
        response = f"I've added {len(order_items)} items to your order. Anything else?"
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM_RESPONSE: {latency_ms:.0f}ms - {response}")
        
        return response
"""
Large Language Model (LLM) reasoning processor for order processing.

This module provides:
- LLMReasoner: Handles menu context and order reasoning logic using Phi-3 Mini via llama.cpp
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama

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
    """Phi-3 Mini reasoning LLM using llama.cpp"""
    
    def __init__(self):
        # Configurable system prompt
        self.system_prompt = "You are an AI fast-food order taker. Respond to the input in a reasonable manner."
        
        # Load model
        model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,  # Context length
            verbose=False
        )
        
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
        """Process user text into order items using Phi-3 Mini"""
        start_time = time.time()
        
        # Create prompt for order processing
        prompt = f"{self.system_prompt}\n\nMenu:\n{self.menu_context}\n\nCustomer: {text}\n\nResponse:"
        
        # Generate response using Phi-3 Mini (run in thread to avoid blocking)
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.llm(prompt, max_tokens=150, stop=["Customer:", "\n\n"])
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        # For now, return mock items - TODO: parse response into actual OrderItems
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM_REASONING: {latency_ms:.0f}ms - '{text}' -> '{response_text}' -> {len(items)} items")
        
        return items
    
    async def generate_response(self, order_items: List[OrderItem]) -> str:
        """Generate response text for TTS using Phi-3 Mini"""
        start_time = time.time()
        
        # Create prompt for response generation
        items_text = ", ".join([f"{item.name} (${item.price})" for item in order_items])
        prompt = f"{self.system_prompt}\n\nI just processed these items: {items_text}\n\nGenerate a brief, friendly response to the customer:\n\nResponse:"
        
        # Generate response using Phi-3 Mini
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.llm(prompt, max_tokens=50, stop=["\n", "Customer:"])
        )
        
        response_text = response['choices'][0]['text'].strip()
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM_RESPONSE: {latency_ms:.0f}ms - {response_text}")
        
        return response_text
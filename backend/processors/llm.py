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
import torch

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

#Template: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
class LLMReasoner:
    """Phi-3 Mini reasoning LLM using llama.cpp"""

    def __init__(self):
        # Check llama-cpp-python CUDA support
        import llama_cpp
        print(f"📋 llama-cpp-python version: {llama_cpp.__version__}")
        
        # Load model with GPU acceleration
        model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"
        print(f"🔧 Loading model with GPU acceleration...")
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,     # Use all GPU layers
            n_ctx=1024,          # Half the context (was 2048)
            n_batch=512,         # GPU batch processing
            n_threads=None,      # Let llama.cpp auto-detect optimal threads
            verbose=True         # Enable to see GPU layer loading
        )
        
        # Check actual GPU layers loaded
        print(f"🎯 Model loaded. Checking GPU configuration...")
        print(f"📊 Context size: {self.llm.n_ctx()}")
        print(f"📊 Vocab size: {self.llm.n_vocab()}")
        
        # Try to detect GPU usage
        try:
            backend_name = getattr(self.llm._model, 'backend_name', 'unknown')
            print(f"📊 Backend: {backend_name}")
        except:
            print("📊 Backend: Could not detect")

        self.menu_context = self.load_menu_context()
        self.current_order = []  # Track current order items

        # Simple model warmup with GPU monitoring
        print("🔥 Warming up model...")
        print("🔍 GPU memory before warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        warmup_start = time.time()
        warmup_response = self.llm("Hello", max_tokens=10)
        warmup_time = (time.time() - warmup_start) * 1000
        
        print("🔍 GPU memory after warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        print(f"🔥 Model warmup: {warmup_time:.0f}ms")
        print(f"🔥 Warmup response: {warmup_response['choices'][0]['text'][:50]}...")

        # Log initial GPU memory usage
        self._log_gpu_memory("LLM_INIT")

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
        """Simple stub - just return mock items"""
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        return items

    async def generate_response(self, text: str) -> str:
        print(f"\n🗣  Test input: '{text}'")

		# self.llm.reset()  # Clears model state and KV cache
        """Fast LLM inference with hardcoded message"""
        start_time = time.time()

        # Generate response using Phi-3 Mini (run in thread to avoid blocking)
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.llm(text, max_tokens=500)
        )

        print(response)
        response_text = response['choices'][0]['text'].strip()

        latency_ms = (time.time() - start_time) * 1000
        print(f"🚀 LLM_INFERENCE: {latency_ms:.0f}ms")

        return response_text

    def _log_gpu_memory(self, context: str):
        """Log GPU memory usage and KV cache stats"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024*1024)
            gpu_reserved = torch.cuda.memory_reserved() / (1024*1024)
            kv_tokens = getattr(self.llm, 'n_tokens', 0)
            kv_max = getattr(self.llm, 'n_ctx', lambda: 2048)()
            kv_size_mb = kv_tokens * 1024 / (1024*1024)  # Rough estimate: 1KB per token

            logger.info(f"GPU_MEMORY_{context}: {gpu_allocated:.1f}MB allocated, {gpu_reserved:.1f}MB reserved | KV_CACHE: {kv_tokens}/{kv_max} tokens (~{kv_size_mb:.1f}MB)")
        else:
            logger.info(f"GPU_MEMORY_{context}: CUDA not available")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = {
            "kv_tokens_used": getattr(self.llm, 'n_tokens', 0),
            "kv_tokens_max": getattr(self.llm, 'n_ctx', lambda: 2048)(),
            "kv_cache_mb_estimate": getattr(self.llm, 'n_tokens', 0) * 1024 / (1024*1024)
        }

        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024*1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024*1024),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024*1024)
            })

        return stats

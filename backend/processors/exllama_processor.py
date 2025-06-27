"""
ExLlamaV2 reasoning processor for order processing.

This module provides:
- ExLlamaV2Reasoner: Alternative to llama-cpp-python using ExLlamaV2
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Cache_8bit
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

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

class ExLlamaV2Reasoner:
    """ExLlamaV2 reasoning LLM for performance comparison"""

    def __init__(self):
        print(f"📋 ExLlamaV2 version: {torch.__version__}")
        
        # Load ExLlamaV2 model
        model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-medium-4k-instruct-exl2-4_25"
        print(f"🔧 Loading ExLlamaV2 model from: {model_path}")
        
        # Initialize ExLlamaV2 config (exact pattern from working test)
        self.config = ExLlamaV2Config()
        self.config.model_dir = str(model_path)
        self.config.prepare()
        
        print(f"📊 Model config loaded:")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Num layers: {self.config.num_hidden_layers}")
        print(f"   Max seq len: {self.config.max_seq_len}")
        
        # Initialize model
        print("🔧 Initializing ExLlamaV2 model...")
        self.model = ExLlamaV2(self.config)
        
        # Check GPU memory before loading
        if torch.cuda.is_available():
            print(f"🔍 GPU memory before model load:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        # Initialize tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        
        # Initialize cache (exact pattern from working test)
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        
        # Load model (exact pattern from working test)
        print("Loading model...")
        load_start = time.time()
        self.model.load_autosplit(self.cache)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
        
        # IMPORTANT: No lazy cache after loading - cache should be ready for streaming
            
        # Initialize generator
        print("🔧 Initializing streaming generator...")
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        
        # Setup greedy sampling to match llama-cpp-python (temperature=0.0, top_k=1)
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.greedy()  # Sets temperature=0.0 for deterministic output
        
        # Check GPU memory after loading
        if torch.cuda.is_available():
            print(f"🔍 GPU memory after model load:")
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        print("✅ ExLlamaV2 model loaded successfully!")

    def _generate_sync(self, prompt: str, max_tokens: int = 500) -> str:
        """Synchronous generation for internal use"""
        try:
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt)
            
            # Generate
            self.generator.warmup()
            self.generator.begin_stream(input_ids, self.settings)
            
            generated_tokens = []
            for _ in range(max_tokens):
                chunk, eos, _ = self.generator.stream()
                if eos or not chunk:
                    break
                generated_tokens.append(chunk)
            
            return "".join(generated_tokens)
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {e}"

    async def generate_response_stream(self, text: str):
        """Generate response using exact working pattern"""
        print(f"\n🗣  Testing: '{text}'")
        
        start_time = time.time()
        
        # Encode input (exact pattern from working test)
        input_ids = self.tokenizer.encode(text)
        print(f"Input tokens: {len(input_ids)}")
        
        # Generate (exact pattern from working test)
        self.generator.warmup()
        self.generator.begin_stream(input_ids, self.settings)
        
        response_tokens = []
        max_tokens = 50
        
        for i in range(max_tokens):
            chunk, eos, _ = self.generator.stream()
            if eos or not chunk:
                break
            response_tokens.append(chunk)
            yield chunk  # Stream the token
            
            # Check if we're taking too long
            elapsed = time.time() - start_time
            if elapsed > 5.0:
                print(f"\n⚠️  TIMEOUT: Generation took more than 5s, stopping...")
                break
        
        total_time = time.time() - start_time
        full_response = "".join(response_tokens)
        
        print(f"\n📊 Results:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Tokens generated: {len(response_tokens)}")
        print(f"   Speed: {len(response_tokens)/total_time:.1f} tokens/sec")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = {
            "backend": "ExLlamaV2",
            "model_path": str(self.config.model_dir),
            "max_seq_len": self.config.max_seq_len,
        }

        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024*1024),
                "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024*1024),
                "gpu_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024*1024)
            })

        return stats
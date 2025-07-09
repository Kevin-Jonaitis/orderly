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
from llama_cpp import Llama, LlamaRAMCache
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
        # ⚠️ DO NOT CHANGE MODEL PATH - Optimized for project requirements
        # model_path = Path(__file__).parent.parent.parent / "models" / "phi-3-mini-4k-instruct-q4_k_s.gguf"  # Q4_K_S (2.0GB) - 30% faster than Q4, potentially, if it has less tokens
        # model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-mini-4k-instruct-q4.gguf"  # Q4 (2.2GB) - slower baseline
        model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-medium-4k-instruct.gguf"  # ⚠️ FIXED MODEL - DO NOT CHANGE
        # model_path = Path(__file__).parent.parent.parent / "models" / "gemma-2b-instruct.gguf"  # Gemma 2B (1.6GB)
        # model_path = Path(__file__).parent.parent.parent / "models" / "Phi-3-mini-4k-instruct-q3_k_s.gguf"  # Q3_K_S (1.7GB) - 55% slower
        print(f"🔧 Loading model with GPU acceleration...")
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,     # Use all GPU layers
            n_ctx=2048,          # ⚠️ FIXED CONTEXT LENGTH - DO NOT CHANGE
            n_batch=512,         # Smaller batch for stability with large contexts
            n_threads=None,      # Let llama.cpp auto-detect optimal threads
            verbose=True,        # Enable to see GPU layer loading
			temperature=0.4, # Greedy decoding for maximum speed
			top_k=1,
            # use_mlock=True,
            flash_attn=True,     # Enable Flash Attention for speedup
        )
        
        # Force clean state - no cache for large prompt testing
        self.llm.reset()
        print(f"🧹 Cache cleared for clean state")
        
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

        self.current_order = []  # Track current order items

        # Perform initial warmup
        self.warmup_with_menu()

        # Log initial GPU memory usage
        self._log_gpu_memory("LLM_INIT")

    def load_instructions_and_menu(self) -> str:
        """Load instructions and menu from separate files"""
        instructions_file = Path("prompts/customer_instructions.txt")
        instructions_content = instructions_file.read_text(encoding='utf-8')
        menu_file = Path("menus/menu_items_descriptions.txt")
        menu_content = menu_file.read_text(encoding='utf-8')
            
        
        # Combine instructions and menu with example conversations
        return f"""{instructions_content}

Menu:
{menu_content}

Now update the order based on the user request below."""

    async def process_order(self, text: str) -> List[OrderItem]:
        """Simple stub - just return mock items"""
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        return items

    def generate_response_stream_for_user_request(self, user_text: str, current_order_summary: str = None, cancellation=None):
        """Generate response for a user request - constructs prompt and calls streaming method"""
        # Use provided order summary or default to empty
        order_section = current_order_summary if current_order_summary else "Previous Order:\n- (empty)"
        
        # Build full prompt with instructions and user input
        full_prompt = f"{self.instructions_and_menu}\n\n{order_section}\n\nUser said: {user_text}\n\n<|end|>\n<|assistant|>"
        
        # Call the core streaming method with the constructed prompt
        return self.generate_response_stream(full_prompt, cancellation)

    def generate_response_stream(self, full_prompt: str, cancellation=None):
        """Generate response with streaming and time-to-first-token metrics"""
        # Add context monitoring for large prompts
        prompt_tokens = len(self.llm.tokenize(full_prompt.encode()))
        context_limit = self.llm.n_ctx()
        max_response_tokens = 500
        
        print(f"📊 Context Analysis:")
        print(f"   Prompt tokens: {prompt_tokens}")
        print(f"   Context limit: {context_limit}")
        print(f"   Current KV cache: {getattr(self.llm, 'n_tokens', 0)} tokens")

        start_time = time.time()
        first_token_time = None
        first_5_words_time = None
        accumulated_text = ""
        word_count = 0
        token_count = 0
        
        # Create streaming response
        stream = self.llm.create_completion(
            full_prompt,
            max_tokens=max_response_tokens,
            stop=["<|user|>", "<|end|>", "User said:"],
            temperature=0.0,
            top_k=1,
            stream=True
        )
        
        for i, output in enumerate(stream):
            
            if 'choices' in output and len(output['choices']) > 0:
                token = output['choices'][0].get('text', '')
                if token:
                    token_count += 1
                    accumulated_text += token
                    
                    # Record time to first token
                    if first_token_time is None:
                        first_token_time = time.time()
                        time_to_first_ms = (first_token_time - start_time) * 1000
                        print(f"⚡ Time to first token: {time_to_first_ms:.0f}ms")
                    
                    # Count words and record time to first 5 words
                    words = accumulated_text.split()
                    if len(words) != word_count:
                        word_count = len(words)
                        if word_count >= 5 and first_5_words_time is None:
                            first_5_words_time = time.time()
                            time_to_5_words_ms = (first_5_words_time - start_time) * 1000
                            print(f"🎯 Time to first 5 words: {time_to_5_words_ms:.0f}ms")
                            print(f"📝 First 5 words: {' '.join(words[:5])}")
                    
                    yield token
        
        # Final metrics
        total_time = time.time() - start_time
        total_ms = total_time * 1000
        tokens_per_second = token_count / total_time if total_time > 0 else 0
        
        print(f"\n📊 Streaming metrics:")
        print(f"   Total tokens: {token_count}")
        print(f"   Total time: {total_ms:.0f}ms")
        print(f"   Speed: {tokens_per_second:.1f} tokens/sec")
        # print(f"   Final response: '{accumulated_text.strip()}'")

    def warmup_with_menu(self):
        """Warm up the model with current instructions and menu"""
        # Reload instructions and menu from file (in case menu was updated)
        self.instructions_and_menu = self.load_instructions_and_menu()
        
        # Realistic model warmup with instructions and menu
        print("🔥 Warming up model with realistic prompt...")
        print("🔍 GPU memory before warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        # Create a realistic warm-up prompt that matches actual usage
        warmup_prompt = f"{self.instructions_and_menu}\n\nPrevious Order:\n- 2x Bean Burrito\n- 2x Taco Supreme\n- 2x Crunchwrap Supreme\n- 1x Pink Lemonade\n\nUser said: hello\n\n<|end|>\n<|assistant|>"
        
        warmup_start = time.time()
        warmup_response = self.llm(warmup_prompt, max_tokens=500, temperature=0.0, top_k=1)
        warmup_time = (time.time() - warmup_start) * 1000
        
        print("🔍 GPU memory after warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        print(f"🔥 Model warmup: {warmup_time:.0f}ms")
        print(f"🔥 Warmup response: {warmup_response['choices'][0]['text'][:100]}...")

        # Synchronize CUDA context to prevent STT conflicts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("🔄 CUDA context synchronized for STT compatibility")

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

    def cancel_generation(self):
        """Cancel ongoing LLM generation using the modified llama-cpp-python cancel() method"""
        self.llm.cancel()
        # print("🚫 LLM generation cancelled via GPU cancellation")
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
from llama_cpp import Llama, LlamaRAMCache, RequestCancellation
import torch

logger = logging.getLogger(__name__)

import threading

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
			temperature=0.0, # Greedy decoding for maximum speed
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

        self.menu_context = self.load_menu_context()
        self.current_order = []  # Track current order items

        # Simple model warmup with GPU monitoring
        print("🔥 Warming up model...")
        print("🔍 GPU memory before warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        warmup_start = time.time()
        warmup_cancellation = RequestCancellation()
        warmup_response = self.llm("Hello", warmup_cancellation, max_tokens=10)
        warmup_time = (time.time() - warmup_start) * 1000
        
        print("🔍 GPU memory after warmup:")
        if torch.cuda.is_available():
            print(f"   Allocated: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
            print(f"   Reserved: {torch.cuda.memory_reserved() / (1024*1024):.1f}MB")
        
        print(f"🔥 Model warmup: {warmup_time:.0f}ms")
        print(f"🔥 Warmup response: {warmup_response['choices'][0]['text'][:50]}...")

        # Synchronize CUDA context to prevent STT conflicts
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print("🔄 CUDA context synchronized for STT compatibility")

        # Log initial GPU memory usage
        self._log_gpu_memory("LLM_INIT")
        
        # Order processing instructions and menu context
        self.instructions_and_menu = """<|user|>
You are a fast-food order taker. Your job is to update the user's order based on their request.

Instructions:
- Always start with a polite human-sounding response.
- Only add/remove/replace items if the user clearly asks.
- Only use items from the menu.
- If the user asks for something off-menu, apologize but do not add it.
- If they want more of an item, increase its count.
- Keep existing items unless removed or replaced.
- Format counts as "- 2x Crunchwrap Supreme".
- Do not include any explanation or suggestions.
- Always reflect the updated order accurately — if you say you're adding something, it must appear in the list.
- If a user asks for an item that could be multiple menu items, ask for clarification, and do not add it to the order.

<|end|>
<|user|>
Previous Order:
- 1x Bean Burrito

User said: can i get a large cheeseburger with fries and a coke actually skip the coke i want a lemonade. and add 2 tacos. oh, and can i also get one more burrito. and a crunchwrap supreme. make that 2 crunchwraps. and can I get a side of sauce?
<|end|>
<|assistant|>
Sorry, we don't serve cheeseburgers or fries. We've added two tacos, another bean burrito, and two Crunchwrap Supremes, and a Pink Lemonade. For sauces, we have several options. Would you like orange sauce, green sauce, or pink sauce?

Updated Order:
- 2x Bean Burrito
- 2x Taco Supreme
- 2x Crunchwrap Supreme
- 1x Pink Lemonade
<|end|>

<|user|>
Previous Order:
- 1x Taco Supreme

User said: can I also get  a bean burrito, and some and a water
<|end|>
<|assistant|>
Absolutely! I added your taco and water. Is there anything else you'd like?

Updated Order:
- 1x Taco Supreme
- 1x Bean Burrito
- 1x Bottled Water
<|end|>

<|user|>


Menu:
Taco Supreme: Ground beef, lettuce, cheddar cheese, diced tomatoes, sour cream, taco shell  
Bean Burrito: Flour tortilla, refried beans, cheddar cheese, red sauce  
Cheesy Gordita Crunch: Flatbread, taco shell, seasoned beef, spicy ranch, lettuce, cheddar cheese  
Crunchwrap Supreme: Flour tortilla, ground beef, nacho cheese, crunchy tostada shell, lettuce, tomato, sour cream
Cheese Quesadilla: Flour tortilla, three-cheese blend, creamy jalapeño sauce
Pink Lemonade: Lemonade with red dye
Tropicana Orange Juice: 100% orange juice
Bottled Water: Purified water
G2 Gatorade Fruit Punch: Electrolyte drink
Frozen Baja Blast: Frozen lime soda slush
Strawberry Skittles Freeze: Frozen drink with Skittles flavor
Nacho Cheese Dip: Melted cheese
Guacamole Dip: Mashed avocado with spices
Pico de Gallo: Chopped tomatoes, onions, cilantro, lime juice
Avocado Ranch Sauce: Creamy ranch with avocado flavor
Creamy Jalapeño Sauce: Spicy, creamy jalapeño blend
Red Sauce: Mild enchilada-style sauce
Fire Sauce Packet: Very spicy sauce
Hot Sauce Packet: Spicy sauce
Mild Sauce Packet: Mildly spicy sauce
Diablo Sauce Packet: Extra spicy sauce
Grilled Chicken Taco: Grilled chicken, lettuce, cheddar cheese, soft tortilla
Double Decker Taco: Crunchy taco with refried beans and soft tortilla
Loaded Nacho Taco: Seasoned beef, nacho cheese, lettuce, red tortilla strips, soft tortilla
Spicy Potato Soft Taco: Seasoned potatoes, lettuce, cheddar cheese, chipotle sauce, soft tortilla
Triple Layer Nachos: Chips, refried beans, red sauce, nacho cheese
Beefy 5-Layer Burrito: Ground beef, nacho cheese, cheddar cheese, refried beans, sour cream, flour tortilla
XXL Grilled Stuft Burrito: Ground beef, rice, beans, guacamole, pico de gallo, cheddar cheese, sour cream

Now update the order based on the user request below."""

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

    def generate_response_stream(self, user_text: str, cancellation: RequestCancellation):
        """Generate response with streaming and time-to-first-token metrics"""
        # Build full prompt with instructions and user input
        full_prompt = f"{self.instructions_and_menu}\n\nPrevious Order:\n- (empty)\n\nUser said: {user_text}\n\n<|end|>\n<|assistant|>"
        
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
        
        # Create streaming response with callback cancellation
        print(f"📍 LLM.generate_response_stream called with cancellation: id={id(cancellation)} at {hex(id(cancellation))}")
        print(f"   Thread ID in generate_response_stream: {threading.current_thread().ident}")
        
        stream = self.llm.create_completion(
            full_prompt,
            max_tokens=max_response_tokens,
            stop=["<|user|>", "<|end|>", "User said:"],
            temperature=0.0,
            top_k=1,
            stream=True,
            should_cancel_callback=cancellation
        )
        
        for i, output in enumerate(stream):
            if i % 10 == 0:  # Check every 10 tokens
                if hasattr(cancellation, 'cancelled'):
                    print(f"\n   Checking cancellation at token {i}: {cancellation.cancelled}")
                else:
                    print(f"\n   Checking cancellation at token {i}: cancellation has no is_cancelled method")
            
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
        print(f"   Final response: '{accumulated_text.strip()}'")

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
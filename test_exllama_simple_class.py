#!/usr/bin/env python3
"""
ExLlamaV2 class using exact working simple test pattern
"""

import time
import torch
import sys
from pathlib import Path

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler

class WorkingExLlamaV2Reasoner:
    """ExLlamaV2 reasoner using exact working pattern"""
    
    def __init__(self):
        print("🔥 Initializing Working ExLlamaV2 Reasoner")
        
        # Load model - exact pattern from working simple test
        model_path = Path("models/exl2_model")
        print(f"Loading model from: {model_path}")
        
        # Initialize config
        self.config = ExLlamaV2Config()
        self.config.model_dir = str(model_path)
        self.config.prepare()
        
        # Initialize model
        self.model = ExLlamaV2(self.config)
        
        # Initialize tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        
        # Initialize cache
        self.cache = ExLlamaV2Cache(self.model, lazy=True)
        
        # Load model
        print("Loading model...")
        load_start = time.time()
        self.model.load_autosplit(self.cache)
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f}s")
        
        # Initialize generator
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        
        # Setup simple sampling
        self.settings = ExLlamaV2Sampler.Settings()
        self.settings.temperature = 0.7
        self.settings.top_k = 50
        self.settings.top_p = 0.9
        
        print("✅ Working ExLlamaV2 initialized successfully!")
    
    async def generate_response_stream(self, text: str):
        """Generate response using exact working pattern"""
        print(f"\n🗣  Testing: '{text}'")
        
        start_time = time.time()
        
        # Encode input
        input_ids = self.tokenizer.encode(text)
        print(f"Input tokens: {len(input_ids)}")
        
        # Generate
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

def test_working_class():
    import asyncio
    
    async def run_test():
        try:
            reasoner = WorkingExLlamaV2Reasoner()
            
            # Test with simple prompt
            simple_prompt = "Hello, how are you today?"
            
            response = ""
            async for token in reasoner.generate_response_stream(simple_prompt):
                response += token
                print(token, end='', flush=True)
            
            print(f"\n\nFinal response: '{response}'")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    success = test_working_class()
    if success:
        print("✅ Working class test successful!")
    else:
        print("❌ Working class test failed!")
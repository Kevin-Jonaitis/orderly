"""
LLM Process
Handles text processing and LLM response generation.
"""

import sys
import os
import asyncio
import multiprocessing
import queue
from datetime import datetime

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.llm import LLMReasoner
from llama_cpp import RequestCancellation

class LLMProcess(multiprocessing.Process):
    """Process that handles LLM text processing and response generation"""
    
    def __init__(self, text_queue):
        super().__init__()
        self.text_queue = text_queue
        
    def run(self):
        """Main process loop - processes text and generates responses"""
        # Initialize LLM processor - let it crash if it fails
        print("🧠 Initializing LLM processor...")
        llm_reasoner = LLMReasoner()
        print("✅ LLM processor loaded")
        print("🧠 LLM process ready for text input...")
        
        # Run async event loop
        asyncio.run(self._process_text_loop(llm_reasoner))
    
    async def _process_text_loop(self, llm_reasoner):
        """Async loop for processing text from STT"""
        last_unique_text = ""
        current_llm_cancellation = None
        
        while True:
            # Get text from STT process - block until new text arrives
            current_text = self.text_queue.get(block=True)
            
            # Text deduplication and processing
            if current_text != last_unique_text:
                print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🎤 NEW TEXT: '{current_text}'")
                last_unique_text = current_text
                
                # Cancel previous LLM request
                if current_llm_cancellation:
                    print("🚫 Cancelling previous LLM request...")
                    current_llm_cancellation.cancel()
                
                # Start new LLM processing - let it crash if it fails
                current_llm_cancellation = RequestCancellation()
                self._stream_response(llm_reasoner, current_text, current_llm_cancellation)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🔄 TEXT UNCHANGED: '{current_text}'")
    
    async def _stream_response(self, llm_reasoner, text, cancellation):
        """Stream LLM response directly to console"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🧠 Processing: '{text}'")
        print("Response: ", end='', flush=True)
        
        # Stream response directly to console - let everything crash
        async for token in llm_reasoner.generate_response_stream(text, cancellation):
            print(token, end='', flush=True)
        
        print(f"\n✅ LLM Complete")
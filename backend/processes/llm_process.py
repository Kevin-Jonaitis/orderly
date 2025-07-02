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
import threading

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.llm import LLMReasoner

class LLMProcess(multiprocessing.Process):
    """Process that handles LLM text processing and response generation"""
    
    def __init__(self, text_queue):
        super().__init__()
        self.text_queue = text_queue
        self.should_cancel = False  # Simple flag for cancellation
        
    def run(self):
        """Main process loop - processes text and generates responses"""
        # Initialize LLM processor - let it crash if it fails
        print("🧠 Initializing LLM processor...")
        llm_reasoner = LLMReasoner()
        print("✅ LLM processor loaded")
        print("🧠 LLM process ready for text input...")
        
        # Run async event loop
        asyncio.run(self._process_text_loop(llm_reasoner))
    
    def _drain_queue(self):
        """Drain the entire queue and return the latest text"""
        # Block for the first item
        latest_text = self.text_queue.get(block=True)
        
        # Then get all remaining items without blocking
        while True:
            try:
                latest_text = self.text_queue.get_nowait()
            except queue.Empty:
                break
        
        return latest_text
    
    async def _process_text_loop(self, llm_reasoner):
        """Async loop for processing text from STT"""
        last_unique_text = ""
        current_task = None
        
        while True:
            # Get all available text from STT process - drain the queue
            current_text = await asyncio.to_thread(self._drain_queue)
            
            # Text deduplication and processing
            if current_text != last_unique_text:
                print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🎤 NEW TEXT: '{current_text}'")
                last_unique_text = current_text
                
                # Cancel previous task if it exists and is still running
                if current_task and not current_task.done():
                    print("🚫 Cancelling previous LLM task...")
                    self.should_cancel = True
                    # Wait for the task to finish
                    import time
                    cancel_start = time.time()
                    try:
                        await current_task
                    except Exception as e:
                        print(f"❌ Task exception: {type(e).__name__}: {e}")
                    cancel_duration = (time.time() - cancel_start) * 1000
                    print(f"✅ Previous task finished in {cancel_duration:.1f}ms")
                
                # Reset flag for new stream
                self.should_cancel = False
                
                # Start new LLM processing asynchronously
                print("Starting new task with text: ", current_text)
                current_task = asyncio.create_task(
                    asyncio.to_thread(self._stream_response, llm_reasoner, current_text)
                )
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🔄 TEXT UNCHANGED: '{current_text}'")
    
    def _stream_response(self, llm_reasoner, text):
        """Stream LLM response directly to console"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🧠 Processing: '{text}'")
        print("Response: ", end='', flush=True)
        
        # Stream response directly to console
        for token in llm_reasoner.generate_response_stream(text, None):
            if self.should_cancel:
                print(f"\n🚫 Cancelled")
                break
            print(token, end='', flush=True)
        
        # Print completion status
        if not self.should_cancel:
            print(f"\n✅ LLM Complete")
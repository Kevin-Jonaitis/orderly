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
import time

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.llm import LLMReasoner

class LLMProcess(multiprocessing.Process):
    """Process that handles LLM text processing and response generation"""
    
    def __init__(self, text_queue, tts_text_queue, llm_start_timestamp, llm_send_to_tts_timestamp, llm_complete_timestamp):
        super().__init__()
        self.text_queue = text_queue
        self.tts_text_queue = tts_text_queue  # Queue to send complete responses to TTS
        self.llm_start_timestamp = llm_start_timestamp
        self.llm_send_to_tts_timestamp = llm_send_to_tts_timestamp
        self.llm_complete_timestamp = llm_complete_timestamp
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
        # Time the queue operation
        queue_start_time = time.time()
        
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
        """Stream LLM response to console AND send partial response to TTS early"""
        # Record timestamp when LLM starts processing
        self.llm_start_timestamp.value = time.time()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🧠 Processing: '{text}'")
        print("Response: ", end='', flush=True)
        
        # Parse tokens and handle early TTS sending
        complete_response, tts_sent = self._parse_and_stream_tokens(llm_reasoner, text)
        
        # Print completion status
        if not self.should_cancel:
            print(f"\n✅ LLM Complete")
    
    def _parse_and_stream_tokens(self, llm_reasoner, text):
        """Parse streaming tokens and send partial response to TTS when 'Updated Order:' is detected"""
        tts_response = ""
        console_response = ""
        tts_sent = False
        llm_start_time = time.time()  # Record start time
        
        for token in llm_reasoner.generate_response_stream(text, None):
            if self.should_cancel:
                break
                
            # Always add to console output
            console_response += token
            print(token, end='', flush=True)
            
            # Accumulate for TTS until we find the marker
            if not tts_sent:
                tts_response += token
                
                # Check for "Updated Order:" marker
                if "Updated Order:" in tts_response:
                    # Set timestamp when sending to TTS
                    self.llm_send_to_tts_timestamp.value = time.time()
                    
                    # Extract text before marker and send to TTS
                    tts_text = tts_response.split("Updated Order:")[0].strip()
                    if tts_text and self.tts_text_queue is not None:
                        self.tts_text_queue.put(tts_text)
                        print(f"\n🎵 Early TTS sent: '{tts_text[:50]}{'...' if len(tts_text) > 50 else ''}'")
                    tts_sent = True
        
        # Set completion timestamp
        self.llm_complete_timestamp.value = time.time()
        
        return console_response, tts_sent
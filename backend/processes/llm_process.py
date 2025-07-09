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
import re
import random

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.llm import LLMReasoner
from utils.order_tracker import OrderTracker

class LLMProcess(multiprocessing.Process):
    """Process that handles LLM text processing and response generation"""
    
    def __init__(self, text_queue, tts_text_queue, llm_start_timestamp, llm_send_to_tts_timestamp, llm_complete_timestamp, order_tracker=None, order_update_queue=None, warmup_queue=None):
        super().__init__()
        self.text_queue = text_queue
        self.tts_text_queue = tts_text_queue  # Queue to send complete responses to TTS
        self.llm_start_timestamp = llm_start_timestamp
        self.llm_send_to_tts_timestamp = llm_send_to_tts_timestamp
        self.llm_complete_timestamp = llm_complete_timestamp
        self.should_cancel = False  # Simple flag for cancellation
        # Use provided order tracker if provided, otherwise create a new one
        self.order_tracker = order_tracker if order_tracker else OrderTracker()
        self.order_update_queue = order_update_queue  # Queue to send order updates to WebSocket
        self.warmup_queue = warmup_queue  # Queue to receive warmup signals
        
        # Alternative phrases for order-taking responses
        self.alternative_phrases = [
            "Got it!",
            "Sure!",
            "Let me make that adjustment for you.",
            "Perfect!",
            "I'll add that for you.",
            "No problem!",
            "You got it!",
            "I'll get that right away.",
            "Consider it done!",
            "I'll update your order.",
            "That sounds great!",
            "I'll make that change.",
            "Got that down!",
            "I'll add that to your order."
        ]
    
    def replace_overused_phrases(self, text):
        """Replace 'Certainly!' and 'Absolutely!' with alternative phrases"""
        # Check if the first sentence starts with these phrases
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            
            # Check for variations of "Certainly" and "Absolutely"
            if (first_sentence.startswith("Certainly") or 
                first_sentence.startswith("Absolutely") or
                first_sentence.startswith("certainly") or
                first_sentence.startswith("absolutely")):
                
                # Remove the overused phrase
                if first_sentence.startswith("Certainly"):
                    first_sentence = first_sentence[10:].strip()  # Remove "Certainly "
                elif first_sentence.startswith("certainly"):
                    first_sentence = first_sentence[10:].strip()  # Remove "certainly "
                elif first_sentence.startswith("Absolutely"):
                    first_sentence = first_sentence[10:].strip()  # Remove "Absolutely "
                elif first_sentence.startswith("absolutely"):
                    first_sentence = first_sentence[10:].strip()  # Remove "absolutely "
                
                # Add a random alternative phrase
                replacement = random.choice(self.alternative_phrases)
                new_first_sentence = f"{replacement} {first_sentence}"
                
                # Reconstruct the text
                sentences[0] = new_first_sentence
                return '. '.join(sentences)
        
        return text
    
    def run(self):
        """Main process loop - processes text and generates responses"""
        # Initialize LLM processor - let it crash if it fails
        print("🧠 Initializing LLM processor...")
        llm_reasoner = LLMReasoner()
        print("✅ LLM processor loaded")
        print("🧠 LLM process ready for text input...")
        
        # Start warmup monitoring thread
        if self.warmup_queue:
            warmup_thread = threading.Thread(target=self._warmup_monitor, args=(llm_reasoner,), daemon=True)
            warmup_thread.start()
            print("🔥 [LLM] Warmup monitoring thread started")
        
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
            try:
                current_text = await asyncio.to_thread(self._drain_queue)
            except Exception as e:
                print(f"❌ [LLM] Error draining queue: {e}")
                current_text = ""
            
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
    
    def _warmup_monitor(self, llm_reasoner):
        """Separate thread to monitor for warmup signals"""
        print("🔥 [LLM] Warmup monitor thread running...")
        while True:
            try:
                # Check for warmup signals (blocking with timeout)
                warmup_signal = self.warmup_queue.get(timeout=1.0)  # 1 second timeout
                print(f"🔥 [LLM] Received signal: {warmup_signal}")
                if warmup_signal == "warmup":
                    print("🔥 [LLM] Received warmup signal, warming up with new menu...")
                    llm_reasoner.warmup_with_menu()
                    print("✅ [LLM] Warmup completed with new menu")
            except queue.Empty:
                continue  # No signal, keep monitoring
            except Exception as e:
                print(f"❌ [LLM] Error in warmup monitor: {e}")
                time.sleep(1)  # Wait before retrying
    
    def _stream_response(self, llm_reasoner, text):
        """Stream LLM response to console AND send partial response to TTS early"""
        # Record timestamp when LLM starts processing
        self.llm_start_timestamp.value = time.time()
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🧠 Processing: '{text}'")
        print("Response: ", end='', flush=True)
        
        # Get current order summary for context
        current_order_summary = self.order_tracker.get_order_summary()
        print(f"📋 Using order context: {current_order_summary}")
        
        # Parse tokens and handle early TTS sending
        complete_response, tts_sent = self._parse_and_stream_tokens(llm_reasoner, text, current_order_summary)
        
        # Print the complete response
        print(f"\n📝 COMPLETE RESPONSE:")
        print("=" * 50)
        print(complete_response)
        print("=" * 50)
        
        # Parse and update order tracker with complete response
        print(f"📋 [Order] Before parsing order tracker state: {self.order_tracker.get_order_summary()}")
        self.order_tracker.parse_and_update_order(complete_response)
        print(f"📋 [Order] After parsing order tracker state: {self.order_tracker.get_order_summary()}")
        
        # Send order update to WebSocket queue if available
        if self.order_update_queue:
            try:
                order_data = self.order_tracker.format_order_for_frontend()
                print(f"📋 [Order] Formatted order data for frontend: {order_data}")
                self.order_update_queue.put(order_data)
                print(f"📋 [Order] Sent formatted update to WebSocket queue: {order_data}")
            except Exception as e:
                print(f"❌ [Order] Error sending update to queue: {e}")
        else:
            print("❌ [Order] No order_update_queue available!")
        
        # Print current order summary
        print(f"\n📋 {self.order_tracker.get_order_summary()}")
        
        # Print completion status
        if not self.should_cancel:
            print(f"\n✅ LLM Complete")
    
    def _parse_and_stream_tokens(self, llm_reasoner, text, current_order_summary):
        """Parse streaming tokens and send response to TTS when 'Updated Order:' is detected, or send full response if not found"""
        tts_response = ""
        response_builder = ""
        tts_sent = False
        llm_start_time = time.time()  # Record start time
        
        for token in llm_reasoner.generate_response_stream_for_user_request(text, current_order_summary):
            if self.should_cancel:
                break
                
            # Always add to console output
            response_builder += token
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
                        # Replace overused phrases before sending to TTS
                        tts_text = self.replace_overused_phrases(tts_text)
                        self.tts_text_queue.put(tts_text)
                        print(f"\n🎵 Early TTS sent: '{tts_text[:50]}{'...' if len(tts_text) > 50 else ''}'")
                    tts_sent = True
        
        # If we didn't find "Updated Order:" by the end, send the complete response to TTS
        if not tts_sent and response_builder.strip():
            # Set timestamp when sending to TTS
            self.llm_send_to_tts_timestamp.value = time.time()
            
            # Send the complete response to TTS
            if self.tts_text_queue is not None:
                # Replace overused phrases before sending to TTS
                tts_text = self.replace_overused_phrases(response_builder.strip())
                self.tts_text_queue.put(tts_text)
                print(f"\n🎵 Complete TTS sent: '{tts_text[:50]}{'...' if len(tts_text) > 50 else ''}'")
            tts_sent = True
        
        # Set completion timestamp
        self.llm_complete_timestamp.value = time.time()
        
        return response_builder, tts_sent
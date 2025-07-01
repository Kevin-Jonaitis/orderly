#!/usr/bin/env python3
"""
Real-time STT → LLM Pipeline
Combines streaming STT with LLM processing, supporting request cancellation and KV cache preservation.
"""

import sys
import os
import time
import asyncio
import argparse
import signal
import numpy as np
import sounddevice as sd
from pathlib import Path

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import processors
from processors.stt import ParakeetSTTProcessor  
from processors.llm import LLMReasoner
import torch
import logging

# Set up minimal logging to reduce noise
logging.getLogger('nemo').setLevel(logging.WARNING)

SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

class RealTimeSTTLLMProcessor:
    """Combined real-time STT → LLM processor with request cancellation and KV cache management"""
    
    def __init__(self):
        print("🚀 Initializing combined STT → LLM pipeline...")
        start_init = time.time()
        
        # Initialize STT processor (reusing the proven implementation)
        print("📝 Loading STT processor...")
        self.stt_processor = self._init_stt_processor()
        print("✅ STT processor loaded successfully")
        
        # Initialize LLM processor
        print("🧠 Loading LLM processor...")
        self.llm_reasoner = LLMReasoner()
        print("✅ LLM processor loaded successfully")
        
        # Request management
        self.current_llm_task = None
        self.accumulated_text = ""
        self.last_unique_text = ""  # Track last unique text to detect new content
        
        # Test input copied exactly from test_llm.py
        self.test_input = """<|user|>
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
- If a user asks for an item that could be multiple menu items, ask for clarification.

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
        
        init_time = (time.time() - start_init) * 1000
        print(f"✅ Combined STT → LLM pipeline initialized in {init_time:.0f}ms")
    
    def _init_stt_processor(self):
        """Initialize STT processor with same config as test_realtime_stt.py"""
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            from omegaconf import open_dict
            
            # Load model exactly like ParakeetSTTProcessor
            model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            
            # Set up autocast for mixed precision
            autocast = torch.amp.autocast(model.device.type, enabled=True)
            
            # Configure decoding
            decoding_cfg = model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"
                decoding_cfg.preserve_alignments = False
                if hasattr(model, 'joint'):
                    decoding_cfg.fused_batch_size = -1
                    if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                        decoding_cfg.greedy.max_symbols = 10
                if hasattr(model, "cur_decoder"):
                    model.change_decoding_strategy(decoding_cfg, decoder_type=model.cur_decoder)
                else:
                    model.change_decoding_strategy(decoding_cfg)
            
            # Model setup
            model = model.cuda()
            model.eval()
            
            # Set attention context size for low latency (80ms)
            ATT_CONTEXT_SIZE = [70, 1]  # 80ms latency
            model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
            
            # Create processor object with all needed attributes
            processor = type('STTProcessor', (), {})()
            processor.model = model
            processor.autocast = autocast
            processor.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer
            
            # Initialize streaming state
            batch_size = 1
            processor.cache_last_channel, processor.cache_last_time, processor.cache_last_channel_len = \
                processor.model.encoder.get_initial_cache_state(batch_size=batch_size)
            processor.previous_hypotheses = None
            processor.pred_out_stream = None
            processor.step_num = 0
            
            # Audio buffering (160ms minimum for stable mel features)
            processor.audio_buffer = []
            processor.min_chunks = 2  # 160ms minimum (2 * 80ms chunks)
            
            return processor
            
        except Exception as e:
            print(f"❌ Failed to initialize STT processor: {e}")
            raise e
    
    
    def downsample_audio(self, audio_chunk):
        """Downsample from 24kHz to 16kHz for NeMo"""
        target_samples = int(len(audio_chunk) * 16000 / 24000)
        indices = np.linspace(0, len(audio_chunk) - 1, target_samples).astype(int)
        return audio_chunk[indices]
    
    async def process_audio_chunk(self, audio_chunk):
        """Process 80ms chunks - buffer minimally to get valid mel features, then send to LLM"""
        try:
            # Downsample to 16kHz for NeMo
            audio_16k = self.downsample_audio(audio_chunk)
            
            # Add to buffer
            self.stt_processor.audio_buffer.extend(audio_16k)
            
            # Process when we have 2 chunks (160ms) for stable mel features
            chunk_samples = len(audio_16k)
            min_samples = chunk_samples * self.stt_processor.min_chunks  # 160ms worth
            
            if len(self.stt_processor.audio_buffer) < min_samples:
                # Need more chunks for stable processing
                return
            
            # Take exactly 2 chunks worth of audio
            audio_to_process = np.array(self.stt_processor.audio_buffer[:min_samples])
            # Remove the processed audio, keep remainder
            self.stt_processor.audio_buffer = self.stt_processor.audio_buffer[chunk_samples:]  # Remove 1 chunk, keep overlap
            
            # Apply preprocessing to match NeMo's expected format  
            audio_tensor = torch.tensor(audio_to_process, dtype=torch.float32).unsqueeze(0)  # (1, time)
            audio_length = torch.tensor([len(audio_to_process)], dtype=torch.long)
            
            # Move tensors to the same device as the model (GPU)
            device = next(self.stt_processor.model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            audio_length = audio_length.to(device)
            
            # Use model's preprocessor to get features
            with torch.no_grad():
                processed_signal, processed_signal_length = self.stt_processor.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_length
                )
            
            # Check if we got valid features
            if processed_signal.numel() == 0 or processed_signal_length.item() == 0:
                return
            
            # Process with streaming model
            start_time = time.time()
            
            with torch.inference_mode():
                with self.stt_processor.autocast:
                    with torch.no_grad():
                        (
                            self.stt_processor.pred_out_stream,
                            transcribed_texts,
                            self.stt_processor.cache_last_channel,
                            self.stt_processor.cache_last_time,
                            self.stt_processor.cache_last_channel_len,
                            self.stt_processor.previous_hypotheses,
                        ) = self.stt_processor.model.conformer_stream_step(
                            processed_signal=processed_signal,
                            processed_signal_length=processed_signal_length,
                            cache_last_channel=self.stt_processor.cache_last_channel,
                            cache_last_time=self.stt_processor.cache_last_time,
                            cache_last_channel_len=self.stt_processor.cache_last_channel_len,
                            keep_all_outputs=True,
                            previous_hypotheses=self.stt_processor.previous_hypotheses,
                            previous_pred_out=self.stt_processor.pred_out_stream,
                            drop_extra_pre_encoded=0 if self.stt_processor.step_num == 0 else self.stt_processor.model.encoder.streaming_cfg.drop_extra_pre_encoded,
                            return_transcription=True,
                        )
            
            process_time = (time.time() - start_time) * 1000
            
            # Extract transcription
            if transcribed_texts and len(transcribed_texts) > 0:
                if hasattr(transcribed_texts[0], 'text'):
                    current_text = transcribed_texts[0].text
                else:
                    current_text = str(transcribed_texts[0])
            else:
                current_text = ""
            
            # Handle STT output and LLM pipeline
            if current_text.strip():
                print(f"🎤 [{self.stt_processor.step_num:3d}] {process_time:3.0f}ms: '{current_text}'")
                
                # Only process if this is actually NEW text
                if current_text.strip() != self.last_unique_text:
                    # Genuinely new text detected
                    self.last_unique_text = current_text.strip()
                    # Replace accumulated text instead of appending (STT gives cumulative results)
                    self.accumulated_text = current_text.strip()
                    
                    # Cancel existing LLM and start new processing immediately
                    await self._schedule_llm_processing()
                # If same text as before, don't cancel LLM
                
            elif self.stt_processor.step_num % 10 == 0:  # Show progress every 10 processed chunks 
                print(f"🔄 [{self.stt_processor.step_num:3d}] Processing...")
            
            self.stt_processor.step_num += 1
            
        except Exception as e:
            print(f"❌ Error processing chunk {self.stt_processor.step_num}: {e}")
    
    async def _schedule_llm_processing(self):
        """Schedule LLM processing immediately when new text is detected"""
        # Cancel any existing LLM task
        if self.current_llm_task and not self.current_llm_task.done():
            print("🚫 Cancelling previous LLM request...")
            # First cancel GPU processing using our new cancel method
            self.llm_reasoner.cancel_generation()
            # Then cancel the asyncio task
            self.current_llm_task.cancel()
            try:
                await self.current_llm_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
        
        # Create new LLM task immediately (no debouncing)
        self.current_llm_task = asyncio.create_task(self._llm_processing())
    
    async def _llm_processing(self):
        """Process accumulated text with LLM immediately"""
        try:
            # Check if we still have text to process
            if not self.accumulated_text.strip():
                return
            
            # Prepare LLM input with proper format from test_llm
            user_input = self.accumulated_text.strip()
            llm_prompt = f"{self.test_input}\n\nPrevious Order:\n- (empty)\n\nUser said: {user_input}\n\n<|end|>\n<|assistant|>"
            
            print(f"\n🧠 Sending to LLM: '{user_input}'")
            
            # Generate streaming response using LLMReasoner
            response_start = time.time()
            
            full_response = ""
            async for token in self.llm_reasoner.generate_response_stream(llm_prompt):
                full_response += token
                # Print streaming output in real-time
                print(token, end='', flush=True)
            
            response_time = (time.time() - response_start) * 1000
            print(f"\n✅ LLM response ({response_time:.0f}ms): '{full_response.strip()}'")
            
            # Clear accumulated text after processing
            self.accumulated_text = ""
            
        except asyncio.CancelledError:
            print("🚫 LLM request cancelled")
            raise
        except Exception as e:
            print(f"❌ Error in LLM processing: {e}")

async def stream_microphone_realtime(device_id=None, show_devices=False):
    """Stream microphone audio with real-time STT → LLM processing"""
    
    if show_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Initialize combined STT → LLM processor
    processor = RealTimeSTTLLMProcessor()
    
    print(f"🎙️  Starting real-time STT → LLM pipeline...")
    print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
    print("🛑 Press Ctrl+C to stop")
    
    audio_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio input - matches Rust server behavior"""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Convert to mono float32 and add to queue (same as Rust server)
        mono_data = indata[:, 0].astype(np.float32).copy()
        loop.call_soon_threadsafe(audio_queue.put_nowait, mono_data)
    
    # Set input device if specified
    if device_id is not None:
        sd.default.device[0] = device_id
    
    # Start audio streaming with same parameters as Rust server
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=CHUNK_SIZE,  # 80ms blocks (1920 samples at 24kHz)
    ):
        try:
            chunk_count = 0
            while True:
                # Get 80ms audio chunk from queue
                audio_chunk = await audio_queue.get()
                chunk_count += 1
                
                # Process each 80ms chunk through STT → LLM pipeline
                await processor.process_audio_chunk(audio_chunk)
                
                # Show processing progress
                if chunk_count % 50 == 0:  # Every ~4 seconds
                    print(f"📊 Processed {chunk_count} chunks ({chunk_count * 80}ms audio)")
                
        except KeyboardInterrupt:
            print("\n🛑 Real-time STT → LLM pipeline stopped")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")

def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupted by user")
    sys.exit(0)

async def main():
    parser = argparse.ArgumentParser(description="Real-time STT → LLM pipeline (80ms chunks)")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Run real-time streaming
    await stream_microphone_realtime(
        device_id=args.device,
        show_devices=args.list_devices
    )

if __name__ == "__main__":
    asyncio.run(main())
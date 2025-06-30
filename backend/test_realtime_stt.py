#!/usr/bin/env python3
"""
Real-time STT test with 80ms chunks - matching Rust server behavior
Uses NeMo's streaming capabilities for true incremental processing
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

# Import NeMo components directly for streaming
import torch
import logging

# Set up minimal logging to reduce noise
logging.getLogger('nemo').setLevel(logging.WARNING)

SAMPLE_RATE = 24000  # Match Rust server (will downsample to 16kHz for NeMo)
CHUNK_DURATION_MS = 80  # 80ms chunks like rust server
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

class RealTimeSTTProcessor:
    """Real-time STT processor using NeMo streaming capabilities"""
    
    def __init__(self):
        print("🚀 Initializing NeMo streaming STT...")
        start_init = time.time()
        
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            from omegaconf import open_dict
            
            # Load model exactly like ParakeetSTTProcessor
            model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            
            # Set up autocast for mixed precision
            self.autocast = torch.amp.autocast(self.model.device.type, enabled=True)
            
            # Configure decoding
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"
                decoding_cfg.preserve_alignments = False
                if hasattr(self.model, 'joint'):
                    decoding_cfg.fused_batch_size = -1
                    if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                        decoding_cfg.greedy.max_symbols = 10
                if hasattr(self.model, "cur_decoder"):
                    self.model.change_decoding_strategy(decoding_cfg, decoder_type=self.model.cur_decoder)
                else:
                    self.model.change_decoding_strategy(decoding_cfg)
            
            # Model setup
            self.model = self.model.cuda()
            self.model.eval()
            
            # Set attention context size for low latency (80ms)
            ATT_CONTEXT_SIZE = [70, 1]  # 80ms latency
            self.model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)
            
            # Store the buffer class for later use
            self.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer
            
            # Initialize streaming state
            batch_size = 1
            self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
                self.model.encoder.get_initial_cache_state(batch_size=batch_size)
            self.previous_hypotheses = None
            self.pred_out_stream = None
            self.step_num = 0
            
            # Minimal buffering - need at least 2 chunks (160ms) for valid mel features
            self.audio_buffer = []
            self.min_chunks = 2  # 160ms minimum (2 * 80ms chunks)
            
            init_time = (time.time() - start_init) * 1000
            print(f"✅ NeMo streaming STT initialized in {init_time:.0f}ms")
            
        except Exception as e:
            print(f"❌ Failed to initialize streaming STT: {e}")
            raise e
    
    def downsample_audio(self, audio_chunk):
        """Downsample from 24kHz to 16kHz for NeMo"""
        # Simple downsampling: take every 3rd sample (24000/16000 = 1.5, approximately)
        # For better quality, could use scipy.signal.resample
        target_samples = int(len(audio_chunk) * 16000 / 24000)
        indices = np.linspace(0, len(audio_chunk) - 1, target_samples).astype(int)
        return audio_chunk[indices]
    
    async def process_audio_chunk(self, audio_chunk):
        """Process 80ms chunks - buffer minimally to get valid mel features"""
        try:
            # Downsample to 16kHz for NeMo
            audio_16k = self.downsample_audio(audio_chunk)
            
            # Add to buffer
            self.audio_buffer.extend(audio_16k)
            
            # Process when we have 2 chunks (160ms) for stable mel features
            chunk_samples = len(audio_16k)
            min_samples = chunk_samples * self.min_chunks  # 160ms worth
            
            if len(self.audio_buffer) < min_samples:
                # Need more chunks for stable processing
                return ""
            
            # Take exactly 2 chunks worth of audio
            audio_to_process = np.array(self.audio_buffer[:min_samples])
            # Remove the processed audio, keep remainder
            self.audio_buffer = self.audio_buffer[chunk_samples:]  # Remove 1 chunk, keep overlap
            
            # Apply preprocessing to match NeMo's expected format  
            audio_tensor = torch.tensor(audio_to_process, dtype=torch.float32).unsqueeze(0)  # (1, time)
            audio_length = torch.tensor([len(audio_to_process)], dtype=torch.long)
            
            # Move tensors to the same device as the model (GPU)
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            audio_length = audio_length.to(device)
            
            # Use model's preprocessor to get features
            with torch.no_grad():
                processed_signal, processed_signal_length = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_length
                )
            
            # Check if we got valid features
            if processed_signal.numel() == 0 or processed_signal_length.item() == 0:
                print(f"⚠️  Still empty features for 160ms chunk, skipping")
                return ""
            
            # Process with streaming model
            start_time = time.time()
            
            with torch.inference_mode():
                with self.autocast:
                    with torch.no_grad():
                        (
                            self.pred_out_stream,
                            transcribed_texts,
                            self.cache_last_channel,
                            self.cache_last_time,
                            self.cache_last_channel_len,
                            self.previous_hypotheses,
                        ) = self.model.conformer_stream_step(
                            processed_signal=processed_signal,
                            processed_signal_length=processed_signal_length,
                            cache_last_channel=self.cache_last_channel,
                            cache_last_time=self.cache_last_time,
                            cache_last_channel_len=self.cache_last_channel_len,
                            keep_all_outputs=True,
                            previous_hypotheses=self.previous_hypotheses,
                            previous_pred_out=self.pred_out_stream,
                            drop_extra_pre_encoded=0 if self.step_num == 0 else self.model.encoder.streaming_cfg.drop_extra_pre_encoded,
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
            
            # Display result if there's new text
            if current_text.strip():
                print(f"🎤 [{self.step_num:3d}] {process_time:3.0f}ms: '{current_text}'")
            elif self.step_num % 10 == 0:  # Show progress every 10 processed chunks 
                print(f"🔄 [{self.step_num:3d}] Processing...")
            
            self.step_num += 1
            return current_text
            
        except Exception as e:
            print(f"❌ Error processing chunk {self.step_num}: {e}")
            return ""

async def stream_microphone_realtime(device_id=None, show_devices=False):
    """Stream microphone audio with real-time 80ms chunk processing"""
    
    if show_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Initialize real-time STT processor
    stt_processor = RealTimeSTTProcessor()
    
    print(f"🎙️  Starting real-time transcription...")
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
                
                # Process each 80ms chunk directly
                result = await stt_processor.process_audio_chunk(audio_chunk)
                
                # Show processing progress
                if chunk_count % 50 == 0:  # Every ~4 seconds
                    print(f"📊 Processed {chunk_count} chunks ({chunk_count * 80}ms audio)")
                
        except KeyboardInterrupt:
            print("\n🛑 Real-time transcription stopped")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")

def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupted by user")
    sys.exit(0)

async def main():
    parser = argparse.ArgumentParser(description="Real-time microphone transcription (80ms chunks)")
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
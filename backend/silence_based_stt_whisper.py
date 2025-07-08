#!/usr/bin/env python3
"""
Silence-based STT using Faster Whisper Tiny
Collects audio until 200ms of silence, then transcribes the complete audio segment.
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
from datetime import datetime
import tempfile
import threading

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Faster Whisper
try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    print("❌ Faster Whisper not installed. Please install with: pip install faster-whisper")
    sys.exit(1)

# Audio configuration
SAMPLE_RATE = 24000  # Input sample rate
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHUNK_DURATION_MS = 20  # 20ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 1920 samples at 24kHz

# Silence detection parameters
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence detection
SILENCE_TIMEOUT_MS = 300  # 200ms of silence triggers transcription

class SilenceBasedSTT:
    """Silence-based STT processor using Faster Whisper Tiny"""
    
    def __init__(self):
        print("🚀 Loading Faster Whisper Tiny model...")
        
        # Load model with GPU support and INT8 for maximum speed
        self.model = WhisperModel("turbo", device="cuda", compute_type="int8_float16")
        
        # Create batched inference pipeline for better performance
        self.batched_model = BatchedInferencePipeline(model=self.model)
        
        # Warm up the model
        self._warm_up_model()
        
        # Initialize audio processing
        self.audio_buffer = []
        self.original_audio_buffer = []  # Keep original 24kHz audio for debug files
        self.silence_duration = 0
        self.segment_count = 0
        
        # Keyboard timing variables
        self.keyboard_stop_time: float | None = None
        self.transcription_start_time: float | None = None
        self.transcription_complete_time: float | None = None
        self.silence_detected_time: float | None = None
        
        # Stats printing flag
        self.stats_printed: bool = False
        
        # Create debug directory
        self.debug_dir = Path("debug_wav_files")
        self.debug_dir.mkdir(exist_ok=True)
        
        print("✅ Silence-based STT initialized successfully")
    
    def _warm_up_model(self):
        """Warm up the model with test_audio.wav"""
        if os.path.exists("test_audio.wav"):
            print("🔥 Warming up model with test_audio.wav...")
            warmup_start = time.time()
            segments, info = self.batched_model.transcribe("test_audio.wav", batch_size=8)
            # Collect all segments
            warmup_text = ""
            for segment in segments:
                warmup_text += segment.text + " "
            warmup_time = (time.time() - warmup_start) * 1000
            
            print(f"🔥 Warm-up transcription: {warmup_text.strip()}")
            print(f"🔥 Warm-up time: {warmup_time:.2f} milliseconds")
        else:
            print("⚠️  test_audio.wav not found, skipping warm-up")
    
    def downsample_audio(self, audio_chunk):
        """Downsample from 24kHz to 16kHz for Whisper"""
        # Simple downsampling: take every 3rd sample (24000/16000 = 1.5)
        target_samples = int(len(audio_chunk) * TARGET_SAMPLE_RATE / SAMPLE_RATE)
        indices = np.linspace(0, len(audio_chunk) - 1, target_samples).astype(int)
        return audio_chunk[indices]
    
    def detect_silence(self, audio_chunk):
        """Detect if audio chunk is silence based on RMS"""
        rms = np.sqrt(np.mean(audio_chunk**2))
        return rms < SILENCE_THRESHOLD
    
    def save_audio_to_wav(self, audio_buffer, filename, sample_rate=TARGET_SAMPLE_RATE):
        """Save audio buffer to WAV file"""
        import soundfile as sf
        
        # Convert float32 [-1, 1] to int16 [-32768, 32767]
        audio_int16 = (np.array(audio_buffer) * 32767).astype(np.int16)
        
        # Save as WAV file
        sf.write(filename, audio_int16, sample_rate)
    
    def process_audio_chunk(self, audio_chunk):
        """Process a single audio chunk and handle silence detection"""
        # Store original audio for debug files
        self.original_audio_buffer.extend(audio_chunk)
        
        # Downsample to 16kHz for processing
        audio_16k = self.downsample_audio(audio_chunk)
        
        # Add to buffer
        self.audio_buffer.extend(audio_16k)
        
        # Check for silence
        if self.detect_silence(audio_chunk):
            self.silence_duration += CHUNK_DURATION_MS
        else:
            self.silence_duration = 0  # Reset silence counter
        
        # Check if we should transcribe
        if self.silence_duration >= SILENCE_TIMEOUT_MS and len(self.audio_buffer) > 0:
            self.silence_detected_time = time.time()
            self._transcribe_buffer()
    
    def _transcribe_buffer(self):
        """Transcribe the current audio buffer"""
        if len(self.audio_buffer) == 0:
            return
        
        buffer_transcribe_start = time.time()  # Start timing for buffer-to-transcription
        self.segment_count += 1
        
        # Record transcription start time
        self.transcription_start_time = time.time()
        
        # Remove the last 200ms of silence from the buffer
        silence_samples = int(SILENCE_TIMEOUT_MS * TARGET_SAMPLE_RATE / 1000)
        audio_without_silence = self.audio_buffer[:-silence_samples] if len(self.audio_buffer) > silence_samples else []
        
        # Also remove silence from original audio buffer for debug files
        original_silence_samples = int(SILENCE_TIMEOUT_MS * SAMPLE_RATE / 1000)
        original_audio_without_silence = self.original_audio_buffer[:-original_silence_samples] if len(self.original_audio_buffer) > original_silence_samples else []
        
        # Check if we have enough audio to transcribe (at least 100ms)
        min_audio_samples = int(100 * TARGET_SAMPLE_RATE / 1000)  # 100ms minimum
        if len(audio_without_silence) < min_audio_samples:
            # print(f"⚠️  Audio too short after silence removal: {len(audio_without_silence)} samples, skipping transcription")
            # Reset buffer and silence counter
            self.audio_buffer = []
            self.original_audio_buffer = []
            self.silence_duration = 0
            return
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_filename = tmp_file.name
        
        try:
            # Save audio to WAV with timing (without the trailing silence)
            save_wav_start = time.time()
            self.save_audio_to_wav(audio_without_silence, temp_filename)
            save_wav_end = time.time()
            save_wav_time = (save_wav_end - save_wav_start) * 1000
            
            # Save debug copy only for completed segments (when keyboard was pressed)
            if self.keyboard_stop_time is not None and not self.stats_printed:
                debug_filename = self.debug_dir / f"completed_segment_{self.segment_count:03d}.wav"
                # Save debug file with original sample rate (24kHz)
                self.save_audio_to_wav(original_audio_without_silence, str(debug_filename), SAMPLE_RATE)
                print(f"💾 Saved debug file: {debug_filename} ({len(original_audio_without_silence)} samples at {SAMPLE_RATE}Hz)")
            
            # Transcribe with timing using Faster Whisper with batched inference
            transcribe_start = time.time()
            segments, info = self.batched_model.transcribe(temp_filename, batch_size=8)
            
            # Collect all segments
            transcribed_text = ""
            for segment in segments:
                transcribed_text += segment.text + " "
            
            transcribe_time = (time.time() - transcribe_start) * 1000
            
            # Record transcription complete time
            self.transcription_complete_time = time.time()
            
            # Calculate audio duration
            audio_duration = len(self.audio_buffer) / TARGET_SAMPLE_RATE
            
            # Calculate keyboard timing if available
            keyboard_timing = None
            silence_to_transcription = None
            keyboard_to_silence = None
            if self.keyboard_stop_time is not None:
                keyboard_timing = (self.transcription_complete_time - self.keyboard_stop_time) * 1000
            if self.silence_detected_time is not None:
                silence_to_transcription = (self.transcription_complete_time - self.silence_detected_time) * 1000
            if self.keyboard_stop_time is not None and self.silence_detected_time is not None:
                keyboard_to_silence = (self.silence_detected_time - self.keyboard_stop_time) * 1000
            
            # Print basic results (always)
            if transcribed_text.strip():
                print(f"\n🎤 Segment {self.segment_count}: '{transcribed_text.strip()}'")
            
            # Print detailed stats if keyboard was pressed and not already printed
            if self.keyboard_stop_time is not None and not self.stats_printed:
                print(f"\n📊 TIMING STATS (Segment {self.segment_count}):")
                if transcribed_text.strip():
                    print(f"🎤 Text: '{transcribed_text.strip()}'")
                if silence_to_transcription is not None:
                    print(f"🔇 Silence to transcription: {silence_to_transcription:.2f}ms")
                print(f" Save audio to WAV: {save_wav_time:.2f} ms")
                print(f"⏱️  Transcribe time: {transcribe_time:.2f} ms")
                print(f"🎵 Audio length: {audio_duration:.2f} seconds")
                print(f"📊 Buffer size: {len(self.audio_buffer)} samples")
                print(f"⌨️  Keyboard to transcription: {keyboard_timing:.2f}ms")
                if keyboard_to_silence is not None:
                    print(f"⌨️  Keyboard to silence: {keyboard_to_silence:.2f}ms")
                print(f"🕒 Transcription complete time: {self.transcription_complete_time:.2f} s")
                if self.silence_detected_time is not None:
                    print(f"🔇 Silence detected time: {self.silence_detected_time:.2f} s")
                print("-" * 50)
                
                # Mark stats as printed
                self.stats_printed = True
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            # Reset buffer and silence counter
            self.audio_buffer = []
            self.original_audio_buffer = []
            self.silence_duration = 0

async def stream_microphone_silence_based(device_id=None, show_devices=False):
    """Stream microphone audio with silence-based transcription"""
    
    if show_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        return
    
    # Initialize silence-based STT
    try:
        stt_processor = SilenceBasedSTT()
    except Exception as e:
        print(f"❌ Failed to initialize STT: {e}")
        if "Faster Whisper not installed" in str(e):
            print("💡 To use silence-based STT, install Faster Whisper with: pip install faster-whisper")
        return
    
    print(f"️  Starting silence-based transcription...")
    print(f"📊 Audio: {SAMPLE_RATE}Hz, {CHUNK_DURATION_MS}ms chunks ({CHUNK_SIZE} samples)")
    print(f" Silence threshold: {SILENCE_THRESHOLD} RMS")
    print(f"⏱️  Silence timeout: {SILENCE_TIMEOUT_MS}ms")
    print("⌨️  Press Enter to mark when you stop talking")
    print("🛑 Press Ctrl+C to stop")
    
    audio_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    
    def audio_callback(indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"⚠️  Audio status: {status}")
        
        # Convert to mono float32 and add to queue
        mono_data = indata[:, 0].astype(np.float32).copy()
        loop.call_soon_threadsafe(audio_queue.put_nowait, mono_data)
    
    def keyboard_input_handler():
        """Handle keyboard input in a separate thread"""
        try:
            while True:
                input()  # Wait for Enter key
                current_time = time.time()
                if stt_processor.keyboard_stop_time is None:
                    stt_processor.keyboard_stop_time = current_time
                    print(f"⌨️  Keyboard stop time recorded at {current_time:.3f}")
                else:
                    print(f"⌨️  Keyboard already recorded at {stt_processor.keyboard_stop_time:.3f}, ignoring duplicate")
        except (EOFError, KeyboardInterrupt):
            pass
    
    # Start keyboard input thread
    keyboard_thread = threading.Thread(target=keyboard_input_handler, daemon=True)
    keyboard_thread.start()
    
    # Set input device if specified
    if device_id is not None:
        sd.default.device = (device_id, sd.default.device[1])
    
    # Start audio streaming
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=CHUNK_SIZE,  # 80ms blocks
    ):
        try:
            chunk_count = 0
            while True:
                # Get 80ms audio chunk from queue
                audio_chunk = await audio_queue.get()
                chunk_count += 1
                
                # Process chunk with silence detection
                stt_processor.process_audio_chunk(audio_chunk)
                
                # Show processing progress
                if chunk_count % 100 == 0:  # Every ~8 seconds
                    print(f"📊 Processed {chunk_count} chunks ({chunk_count * 80}ms audio)")
                
        except KeyboardInterrupt:
            print("\n🛑 Silence-based transcription stopped")
        except Exception as e:
            print(f"❌ Error during streaming: {e}")

def handle_sigint(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Interrupted by user")
    sys.exit(0)

async def main():
    parser = argparse.ArgumentParser(description="Silence-based microphone transcription with Faster Whisper Tiny")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )
    
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Run silence-based streaming
    await stream_microphone_silence_based(
        device_id=args.device,
        show_devices=args.list_devices
    )

if __name__ == "__main__":
    asyncio.run(main()) 
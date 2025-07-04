#!/usr/bin/env python3

import time as time_module
import multiprocessing
from multiprocessing import Process
import sounddevice as sd
import numpy as np
import pickle
import queue


class AudioProcessor(Process):
    """Dedicated audio processing - handles all audio streaming operations
    
    This process eliminates the threading complexity from TTSProcess and provides
    a clean separation of concerns. Audio handling is modeled after the working
    test_tts_audio_minimal.py structure for optimal performance.
    """
    
    def __init__(self, audio_queue, first_audio_chunk_timestamp, use_blocking_audio=False, debug_mode=False):
        super().__init__(name="AudioProcess")
        self.audio_queue = audio_queue  # multiprocessing.Queue
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.use_blocking_audio = use_blocking_audio
        self.debug_mode = debug_mode  # True for standalone testing, False for production pipeline
        self.prerecorded_chunks = None
        self.audio_stream = None
        self.first_playback = True
        
    def _load_debug_chunks(self):
        """Load pre-recorded chunks for debugging (same as TTSProcess)"""
        try:
            self.prerecorded_chunks = pickle.load(open('debug_chunks_1751577529.pkl', 'rb'))
            print(f"🔍 [AudioProcessor] Loaded {len(self.prerecorded_chunks)} pre-recorded chunks")
        except Exception as e:
            print(f"❌ [AudioProcessor] Failed to load debug chunks: {e}")
            self.debug_mode = False
            
    def _queue_debug_chunks(self):
        """Queue debug chunks for testing (matches test_tts_audio_minimal pattern)"""
        if not self.prerecorded_chunks:
            return
            
        print(f"🎵 [AudioProcessor] Queuing {len(self.prerecorded_chunks)} debug chunks...")
        start_time = time_module.time()
        
        # Queue all chunks immediately (like test_tts_audio_minimal)
        for i, chunk in enumerate(self.prerecorded_chunks):
            self.audio_queue.put(chunk)
            if i == 0:
                first_queue_time = time_module.time()
                print(f"🎵 [AudioProcessor] First chunk queued at: {first_queue_time:.3f}")
        
        end_time = time_module.time()
        queue_duration = (end_time - start_time) * 1000
        print(f"🎵 [AudioProcessor] All chunks queued in {queue_duration:.1f}ms")
        
        # Set timestamp for timing measurements
        self.first_audio_chunk_timestamp.value = time_module.time()
        print(f"🎵 [AudioProcessor] FIRST CHUNK READY: {self.first_audio_chunk_timestamp.value:.3f}")
    
    def run(self):
        """Main audio process - simple structure like test_tts_audio_minimal"""
        print("🎵 [AudioProcessor] Starting audio processing...")
        
        if self.debug_mode:
            print("🔧 [AudioProcessor] Loading debug chunks...")
            self._load_debug_chunks()
            self._queue_debug_chunks()
            print("✅ [AudioProcessor] Debug chunks loaded and queued")
        
        # Handle audio with minimal complexity
        self._handle_audio()
        
    def _handle_audio(self):
        """Direct audio handling - matches test_tts_audio_minimal structure"""
        print(f"🎵 [AudioProcessor] Starting audio handling (mode: {'blocking' if self.use_blocking_audio else 'callback'})")
        
        if self.use_blocking_audio:
            self._blocking_audio()
        else:
            self._callback_audio()
    
    def _blocking_audio(self):
        """Blocking audio implementation"""
        try:
            print("🎵 [AudioProcessor] Using blocking write mode")
            
            # Create OutputStream without callback
            self.audio_stream = sd.OutputStream(
                samplerate=24000,
                channels=1,
                dtype='float32',
                blocksize=2048,
                latency=0.001
            )
            
            print(f"🔊 [AudioProcessor] Stream config: blocksize={self.audio_stream.blocksize}, latency={self.audio_stream.latency}")
            
            # Start stream after chunks are ready
            self.audio_stream.start()
            print("▶️  [AudioProcessor] Stream started...")
            
            # Simple blocking write loop
            while True:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Record first write timing
                    if self.first_playback:
                        print(f"🔊 [AudioProcessor] FIRST WRITE: {time_module.time():.3f}")
                        self.first_playback = False
                    
                    # Always flatten to ensure correct shape
                    audio_chunk = audio_chunk.flatten()
                    
                    # Direct write
                    self.audio_stream.write(audio_chunk)
                    
                except queue.Empty:
                    # No audio available - continue waiting
                    continue
                except Exception as e:
                    print(f"❌ [AudioProcessor] Error in blocking write: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ [AudioProcessor] Error in blocking audio: {e}")
            import traceback
            traceback.print_exc()
    
    def _callback_audio(self):
        """Callback audio implementation - matches test_tts_audio_minimal"""
        try:
            print("🎵 [AudioProcessor] Using callback mode")
            
            # Track callbacks like test_tts_audio_minimal
            callback_count = 0
            first_callback_time = None
            
            def audio_callback(outdata, frames, time, status):
                """Audio callback - matches test_tts_audio_minimal structure"""
                nonlocal callback_count, first_callback_time
                
                callback_count += 1
                current_time = time_module.time()
                
                if first_callback_time is None:
                    first_callback_time = current_time
                    print(f"🔊 [AudioProcessor] FIRST CALLBACK: {current_time:.3f}")
                
                try:
                    # Use get_nowait like test_tts_audio_minimal
					# TODO: THIS NEEDS TO BE FIXED IF WERE ACTUALLY GOING TO USE THIS
                    audio_chunk = self.audio_queue.get_nowait()
                    
                    # Record first playback timing
                    if self.first_playback:
                        print(f"🔊 [AudioProcessor] First chunk consumed at: {current_time:.3f}")
                        self.first_playback = False
                    
                    # Always flatten to ensure correct shape
                    audio_chunk = audio_chunk.flatten()
                    outdata[:, 0] = audio_chunk
                    
                    print(f"🔊 [AudioProcessor] Callback {callback_count}: consumed chunk at {current_time:.3f}")
                    
                except:
                    # No audio ready - fill with silence
                    outdata.fill(0)
                    print(f"🔇 [AudioProcessor] Callback {callback_count}: queue empty at {current_time:.3f}")
            
            # Create OutputStream with callback
            self.audio_stream = sd.OutputStream(
                callback=audio_callback,
                samplerate=24000,
                channels=1,
                dtype='float32',
                blocksize=2048,
                latency=0.001
            )
            
            print(f"🔊 [AudioProcessor] Stream config: blocksize={self.audio_stream.blocksize}, latency={self.audio_stream.latency}")
            
            # Start stream after chunks are ready (critical for low latency)
            self.audio_stream.start()
            print("▶️  [AudioProcessor] Stream started with chunks ready...")
            
            # Keep process alive without blocking loops (unlike old TTSProcess)
            try:
                while True:
                    time_module.sleep(0.1)  # Light sleep to keep process alive
            except KeyboardInterrupt:
                print("🛑 [AudioProcessor] Stopping...")
                
        except Exception as e:
            print(f"❌ [AudioProcessor] Error in callback audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
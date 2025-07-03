#!/usr/bin/env python3

import time as time_module
import threading
import queue
from multiprocessing import Process
import sounddevice as sd
import numpy as np
import soundfile as sf
from processors.orpheus_tts import OrpheusTTS


class TTSProcess(Process):
    """TTS process that receives text from LLM and streams audio to speakers.
    
    Modeled after test_tts.py with real-time audio streaming.
    Runs in a separate process to avoid blocking the LLM.
    """
    
    def __init__(self, tts_text_queue, first_audio_chunk_timestamp):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.tts = None
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.debug_mode = True  # Set to False to use real TTS generation
        self.use_blocking_audio = False  # Set to False to use callback mode
        self.prerecorded_chunks = None  # Will load during init
    
    def _load_debug_chunks(self):
        """Load pre-recorded chunks for debugging"""
        try:
            import pickle
            self.prerecorded_chunks = pickle.load(open('debug_chunks_1751577529.pkl', 'rb'))
            print(f"🔍 [DEBUG] Loaded {len(self.prerecorded_chunks)} pre-recorded chunks")
        except Exception as e:
            print(f"❌ [DEBUG] Failed to load debug chunks: {e}")
            self.debug_mode = False  # Fall back to real TTS
    
    def _process_debug_chunks(self, audio_queue):
        """Process pre-recorded chunks - optimized for speed"""
        start_time = time_module.time()
        print(f"🔍 [DEBUG] Starting chunk processing at {start_time:.3f}")
        
        # Simple, fast queuing - no expensive operations
        for i, audio_array in enumerate(self.prerecorded_chunks):
            # Direct queue - no splitting since blocksize now matches chunk size (2048)
            audio_queue.put(audio_array)
        
        end_time = time_module.time()
        duration = (end_time - start_time) * 1000
        print(f"🔍 [DEBUG] Queued {len(self.prerecorded_chunks)} chunks in {duration:.1f}ms at {end_time:.3f}")
    
    def _audio_thread(self, audio_queue):
        """Dedicated thread for audio operations - switches between blocking/callback modes"""
        print(f"🎵 [Audio Thread] Starting audio operations (mode: {'blocking' if self.use_blocking_audio else 'callback'})...")
        
        if self.use_blocking_audio:
            self._blocking_audio_loop(audio_queue)
        else:
            self._callback_audio_loop(audio_queue)
    
    def _blocking_audio_loop(self, audio_queue):
        """Blocking write audio implementation"""
        try:
            print("🎵 [Audio] Using blocking write mode")
            
            # Create OutputStream without callback
            self.audio_stream = sd.OutputStream(
                samplerate=24000, 
                channels=1, 
                dtype='float32',
                blocksize=2048,  # Match TTS chunk size for efficiency
                latency=0.001   # Minimal latency
            )
            
            print(f"🔍 [Audio] OutputStream configuration:")
            print(f"   Samplerate: {self.audio_stream.samplerate}")
            print(f"   Blocksize: {self.audio_stream.blocksize}")
            print(f"   Latency: {self.audio_stream.latency}")
            print(f"   Device: {self.audio_stream.device}")
            
            # Start the audio stream
            self.audio_stream.start()
            print("🎵 [Audio] Stream started and ready")
            
            # Simple blocking write loop
            while True:
                # Time the queue wait
                queue_start = time_module.time()
                audio_chunk = audio_queue.get()  # Block until audio available
                queue_end = time_module.time()
                queue_wait_ms = (queue_end - queue_start) * 1000
                print(f"🔍 Queue wait: {queue_wait_ms:.1f}ms")
                
                # Record first write timing
                if self.first_playback:
                    print(f"🔊 FIRST WRITE: {queue_end:.3f}")
                    self.first_playback = False
                
                # Always flatten to ensure correct shape
                audio_chunk = audio_chunk.flatten()
                
                # Time the audio write
                write_start = time_module.time()
                self.audio_stream.write(audio_chunk)
                write_end = time_module.time()
                write_duration_ms = (write_end - write_start) * 1000
                print(f"🔍 Write duration: {write_duration_ms:.1f}ms")
            
        except Exception as e:
            print(f"❌ [Audio] Error in blocking audio: {e}")
            import traceback
            traceback.print_exc()
    
    def _callback_audio_loop(self, audio_queue):
        """Callback audio implementation"""
        try:
            print("🎵 [Audio] Using callback mode")
            
            def audio_callback(outdata, frames, time, status):
                """Audio callback - called by audio system when it needs data"""
                if not self.audio_active:
                    outdata.fill(0)
                    return
                
                try:
                    # Time the queue operation
                    queue_start = time_module.time()
                    audio_chunk = audio_queue.get()
                    queue_end = time_module.time()
                    queue_wait_ms = (queue_end - queue_start) * 1000
                    print(f"🔍 Callback queue wait: {queue_wait_ms:.1f}ms")
                    
                    # Record first playback timing
                    if self.first_playback:
                        print(f"🔊 FIRST CALLBACK: {queue_end:.3f}")
                        self.first_playback = False
                    
                    # Flatten and assign to output
                    audio_chunk = audio_chunk.flatten()
                    outdata[:, 0] = audio_chunk
                    
                except queue.Empty:
                    # No audio ready - fill with silence
                    outdata.fill(0)
                print("OUTDATA: ", outdata)
            
            # Create OutputStream with callback
            self.audio_stream = sd.OutputStream(
                callback=audio_callback,
                samplerate=24000, 
                channels=1, 
                dtype='float32',
                blocksize=2048,  # Match TTS chunk size for efficiency
                latency=0.001   # Minimal latency
            )
            
            print(f"🔍 [Audio] OutputStream configuration:")
            print(f"   Samplerate: {self.audio_stream.samplerate}")
            print(f"   Blocksize: {self.audio_stream.blocksize}")
            print(f"   Latency: {self.audio_stream.latency}")
            print(f"   Device: {self.audio_stream.device}")
            
            # Start the audio stream
            self.audio_stream.start()
            print("🎵 [Audio] Stream started and ready")
            
            # Keep callback thread alive
            while True:
                time_module.sleep(1)
            
        except Exception as e:
            print(f"❌ [Audio] Error in callback audio: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main TTS process - coordinate audio and TTS threads"""
        print("🎵 Starting TTS process with complete threading...")
        
        # Initialize TTS system
        if not self.debug_mode:
            print("🔧 Initializing OrpheusTTS...")
            self.tts = OrpheusTTS()
            print("✅ TTS initialization complete")
        else:
            print("🔧 Loading debug chunks...")
            self._load_debug_chunks()
            print("✅ Debug chunks loaded")
        
        # Set up shared resources
        audio_queue = queue.Queue(maxsize=100)
        self.first_playback = True
        self.audio_active = True  # Simple boolean instead of Event
        
        # Start audio thread
        audio_thread = threading.Thread(
            target=self._audio_thread,
            args=(audio_queue,),
            name="TTS-Audio-Thread"
        )
        audio_thread.daemon = True
        audio_thread.start()
        
        # Wait for audio stream to be ready
        time_module.sleep(0.1)
        
        # Start TTS thread
        tts_thread = threading.Thread(
            target=self._tts_main_thread,
            args=(audio_queue,),
            name="TTS-Main-Thread"
        )
        tts_thread.daemon = True
        tts_thread.start()
        
        # Just wait for TTS thread (daemon threads auto-cleanup on exit)
        try:
            tts_thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt - process will exit")
        
        print("🛑 TTS process complete")
    
    def _tts_main_thread(self, audio_queue):
        """Dedicated thread for TTS operations - handles while loop and generation"""
        try:
            print("🎧 [TTS] Thread ready - waiting for text from LLM...")
            
            while True:
                # Block until we receive text from LLM process
                text = self.tts_text_queue.get()
                
                # Check for shutdown signal
                if text is None:
                    print("🛑 [TTS] Received shutdown signal")
                    break
                
                print(f"📥 [TTS] RECEIVED TEXT: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Reset state for new generation
                self.first_playback = True
                
                if self.debug_mode:
                    # Debug path: use pre-recorded chunks (near-instantaneous)
                    print(f"🔍 [DEBUG] Using pre-recorded chunks")
                    self.first_audio_chunk_timestamp.value = time_module.time()
                    print(f"🎵 [DEBUG] FIRST CHUNK READY: {self.first_audio_chunk_timestamp.value:.3f}")
                    
                    # Signal that audio is ready
                    self.audio_active = True
                    
                    # Process pre-recorded chunks
                    self._process_debug_chunks(audio_queue)
                    
                else:
                    # Production path: real TTS generation
                    first_chunk = True
                    print(f"🎵 [TTS] GENERATION STARTED")
                    
                    debug_audio_chunks = []
                    
                    for result in self.tts.tts_streaming(text, "tara", save_file=None):
                        sample_rate, audio_array, chunk_count = result
                        
                        # Collect for debug
                        debug_audio_chunks.append(audio_array.copy())
                        
                        if first_chunk:
                            # Set the timestamp when first audio chunk is ready
                            self.first_audio_chunk_timestamp.value = time_module.time()
                            print(f"🎵 [TTS] FIRST CHUNK READY: {self.first_audio_chunk_timestamp.value:.3f}")
                            
                            # Analyze first chunk
                            chunk_max = np.max(np.abs(audio_array))
                            chunk_rms = np.sqrt(np.mean(audio_array**2))
                            print(f"🔍 [TTS] First chunk - Max: {chunk_max:.4f}, RMS: {chunk_rms:.4f}")
                            
                            first_chunk = False
                            
                            # Signal that audio is ready
                            self.audio_active = True
                        
                        # Split into 512-sample blocks
                        chunk_size = 512
                        audio_flat = audio_array.flatten()
                        
                        # Queue blocks
                        num_blocks = len(audio_flat) // chunk_size
                        for i in range(num_blocks):
                            start_idx = i * chunk_size
                            end_idx = start_idx + chunk_size
                            audio_block = audio_flat[start_idx:end_idx]
                            audio_queue.put(audio_block)
                        
                        # Handle remaining samples
                        remaining_samples = len(audio_flat) % chunk_size
                        if remaining_samples > 0:
                            remaining_audio = audio_flat[num_blocks * chunk_size:]
                            padded_audio = np.pad(remaining_audio, (0, chunk_size - remaining_samples), mode='constant')
                            audio_queue.put(padded_audio)
                    
                    print("🎵 [TTS] Generation complete")
                    
                    # Save debug audio
                    if debug_audio_chunks:
                        # Also save full audio
                        self._save_debug_audio(debug_audio_chunks, sample_rate)
                
                
        except Exception as e:
            print(f"❌ [TTS] Error in TTS thread: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_debug_audio(self, audio_chunks, sample_rate):
        """Save debug audio (moved from inline)"""
        try:
            flattened_chunks = [chunk.flatten() for chunk in audio_chunks]
            complete_audio = np.concatenate(flattened_chunks, axis=0)
            
            if len(complete_audio.shape) > 1:
                complete_audio = complete_audio.flatten()
            
            debug_filename = f"debug_tts_audio_{int(time_module.time())}.wav"
            sf.write(debug_filename, complete_audio, sample_rate, format='WAV', subtype='PCM_16')
            print(f"🔍 Audio saved to: {debug_filename}")
            
            # Analyze complete audio
            total_samples = len(complete_audio)
            total_duration_ms = (total_samples / sample_rate) * 1000
            print(f"🔍 Audio duration: {total_duration_ms:.1f}ms ({total_samples} samples)")
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
    

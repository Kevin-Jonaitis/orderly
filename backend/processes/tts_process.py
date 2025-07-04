#!/usr/bin/env python3

import time as time_module
from multiprocessing import Process
import numpy as np
import soundfile as sf
from processors.orpheus_tts import OrpheusTTS


class TTSProcess(Process):
    """TTS process that generates audio and sends chunks to AudioProcessor.
    
    Simplified design that focuses only on TTS generation.
    Audio handling is delegated to AudioProcessor for better separation.
    """
    
    def __init__(self, tts_text_queue, audio_queue, first_audio_chunk_timestamp):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.audio_queue = audio_queue  # multiprocessing.Queue to AudioProcessor
        self.tts = None
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.debug_mode = True  # Set to False to use real TTS generation
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
    
    def _process_debug_chunks(self):
        """Process pre-recorded chunks - send to AudioProcessor"""
        start_time = time_module.time()
        print(f"🔍 [TTS] Starting chunk processing at {start_time:.3f}")
        
        # Simple, fast queuing to AudioProcessor
        for i, audio_array in enumerate(self.prerecorded_chunks):
            # Direct queue to AudioProcessor
            self.audio_queue.put(audio_array)
        
        end_time = time_module.time()
        duration = (end_time - start_time) * 1000
        print(f"🔍 [TTS] Queued {len(self.prerecorded_chunks)} chunks to AudioProcessor in {duration:.1f}ms at {end_time:.3f}")
    
    # Audio handling removed - now handled by AudioProcessor
    
    def run(self):
        """Main TTS process - simplified without audio threading"""
        print("🎵 Starting TTS process (audio handled by AudioProcessor)...")
        
        # Initialize TTS system
        if not self.debug_mode:
            print("🔧 Initializing OrpheusTTS...")
            self.tts = OrpheusTTS()
            print("✅ TTS initialization complete")
        else:
            print("🔧 Loading debug chunks...")
            self._load_debug_chunks()
            print("✅ Debug chunks loaded")
        
        # Run TTS processing directly (no threading complexity)
        try:
            self._tts_main_loop()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt - process will exit")
        
        print("🛑 TTS process complete")
    
    def _tts_main_loop(self):
        """Main TTS processing loop - sends chunks to AudioProcessor"""
        try:
            print("🎧 [TTS] Ready - waiting for text from LLM...")
            
            while True:
                # Block until we receive text from LLM process
                text = self.tts_text_queue.get()
                
                # Check for shutdown signal
                if text is None:
                    print("🛑 [TTS] Received shutdown signal")
                    break
                
                print(f"📥 [TTS] RECEIVED TEXT: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                if self.debug_mode:
                    # Debug path: use pre-recorded chunks
                    print(f"🔍 [TTS] Using pre-recorded chunks")
                    self.first_audio_chunk_timestamp.value = time_module.time()
                    print(f"🎵 [TTS] FIRST CHUNK READY: {self.first_audio_chunk_timestamp.value:.3f}")
                    
                    # Send pre-recorded chunks to AudioProcessor
                    self._process_debug_chunks()
                    
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
                        
                        # Split into blocks and send to AudioProcessor
                        chunk_size = 512
                        audio_flat = audio_array.flatten()
                        
                        # Send blocks to AudioProcessor
                        num_blocks = len(audio_flat) // chunk_size
                        for i in range(num_blocks):
                            start_idx = i * chunk_size
                            end_idx = start_idx + chunk_size
                            audio_block = audio_flat[start_idx:end_idx]
                            self.audio_queue.put(audio_block)
                        
                        # Handle remaining samples
                        remaining_samples = len(audio_flat) % chunk_size
                        if remaining_samples > 0:
                            remaining_audio = audio_flat[num_blocks * chunk_size:]
                            padded_audio = np.pad(remaining_audio, (0, chunk_size - remaining_samples), mode='constant')
                            self.audio_queue.put(padded_audio)
                    
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
    

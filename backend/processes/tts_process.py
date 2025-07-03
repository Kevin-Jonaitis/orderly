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
    
    def __init__(self, tts_text_queue, stt_process_time, stt_total_chunk_time, audio_capture_delay,
                 last_word_processing_time, last_text_change_time, llm_queue_wait_time, llm_to_tts_time, llm_total_time, 
                 timestamp_llm_start, timestamp_llm_complete, timestamp_tts_start, timestamp_first_audio,
                 manual_speech_end_time, manual_audio_heard_time):
        super().__init__(name="TTSProcess")
        self.tts_text_queue = tts_text_queue
        self.tts = None
        # All timing data for latency tracking
        self.stt_process_time = stt_process_time            # STT internal processing time in ms
        self.stt_total_chunk_time = stt_total_chunk_time    # STT complete function time in ms
        self.audio_capture_delay = audio_capture_delay      # Audio capture to processing delay in ms
        self.last_word_processing_time = last_word_processing_time  # Frame-to-text time in ms
        self.last_text_change_time = last_text_change_time  # When STT text last changed
        self.llm_queue_wait_time = llm_queue_wait_time      # LLM queue waiting time in ms
        self.llm_to_tts_time = llm_to_tts_time              # LLM time-to-TTS in ms
        self.llm_total_time = llm_total_time                # Total LLM time in ms
        self.timestamp_llm_start = timestamp_llm_start
        self.timestamp_llm_complete = timestamp_llm_complete
        self.timestamp_tts_start = timestamp_tts_start
        self.timestamp_first_audio = timestamp_first_audio
        self.timestamp_audio_playback_start = 0.0  # When audio actually starts playing
        # Manual timing data
        self.manual_speech_end_time = manual_speech_end_time    # When user stops talking
        self.manual_audio_heard_time = manual_audio_heard_time  # When user hears audio
    
    
    def _audio_thread(self, audio_queue):
        """Dedicated thread for audio operations to prevent GIL blocking"""
        try:
            print("🎵 [Audio Thread] Starting audio operations...")
            
            def audio_callback(outdata, frames, time, status):
                """Non-blocking audio callback - called by audio system when it needs data"""
                if not self.audio_active:
                    outdata.fill(0)
                    return
                
                try:
                    audio_chunk = audio_queue.get_nowait()
                    
                    # Record first playback timing
                    if self.first_playback:
                        timestamp_audio_playback_start = time_module.time()
                        self.timestamp_audio_playback_start = timestamp_audio_playback_start
                        print(f"🔊 [Audio] PLAYBACK START: {timestamp_audio_playback_start:.3f}")
                        self.first_playback = False
                    
                    # Output audio
                    outdata[:, 0] = audio_chunk
                    
                except queue.Empty:
                    # No audio ready - fill with silence
                    outdata.fill(0)
            
            # Create OutputStream in the audio thread
            self.audio_stream = sd.OutputStream(
                callback=audio_callback,
                samplerate=24000, 
                channels=1, 
                dtype='float32',
                blocksize=512,  # Fixed 512-sample blocks
                latency=0.002   # 2ms latency
            )
            
            print(f"🔍 [Audio] OutputStream configuration:")
            print(f"   Samplerate: {self.audio_stream.samplerate}")
            print(f"   Blocksize: {self.audio_stream.blocksize}")
            print(f"   Latency: {self.audio_stream.latency}")
            print(f"   Device: {self.audio_stream.device}")
            
            # Start the audio stream
            self.audio_stream.start()
            print("🎵 [Audio] Stream started and ready")
            
            # Keep audio thread alive (daemon thread will be killed on process exit)
            try:
                while True:
                    time_module.sleep(1)
            except:
                pass
            
        except Exception as e:
            print(f"❌ [Audio] Error in audio thread: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main TTS process - coordinate audio and TTS threads"""
        print("🎵 Starting TTS process with complete threading...")
        
        # Initialize TTS system
        print("🔧 Initializing OrpheusTTS...")
        self.tts = OrpheusTTS()
        print("✅ TTS initialization complete")
        
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
                
                # Record timestamp when TTS receives text from LLM
                tts_receive_time = time_module.time()
                self.timestamp_tts_start.value = tts_receive_time
                print(f"📥 [TTS] RECEIVED TEXT: {tts_receive_time:.3f}")
                print(f"🎤 [TTS] Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Reset state for new generation
                self.first_playback = True
                
                # Generate TTS audio directly in this thread
                first_chunk = True
                tts_generation_start = time_module.time()
                print(f"🎵 [TTS] GENERATION STARTED: {tts_generation_start:.3f}")
                
                debug_audio_chunks = []
                
                for result in self.tts.tts_streaming(text, "tara", save_file=None):
                    sample_rate, audio_array, chunk_count = result
                    
                    # Collect for debug
                    debug_audio_chunks.append(audio_array.copy())
                    
                    if first_chunk:
                        chunk_ready_time = time_module.time()
                        self.timestamp_first_audio.value = chunk_ready_time
                        print(f"🎵 [TTS] FIRST CHUNK READY: {chunk_ready_time:.3f}")
                        
                        # Analyze first chunk
                        chunk_max = np.max(np.abs(audio_array))
                        chunk_rms = np.sqrt(np.mean(audio_array**2))
                        print(f"🔍 [TTS] First chunk - Max: {chunk_max:.4f}, RMS: {chunk_rms:.4f}")
                        
                        first_chunk = False
                        chunk_generation_time = (chunk_ready_time - tts_generation_start) * 1000
                        print(f"🎵 [TTS] First chunk after {chunk_generation_time:.1f}ms")
                        
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
                    # Save chunks for future replay
                    import pickle
                    chunk_filename = f'debug_chunks_{int(time_module.time())}.pkl'
                    pickle.dump(debug_audio_chunks, open(chunk_filename, 'wb'))
                    print(f"🔍 [TTS] Saved {len(debug_audio_chunks)} chunks to {chunk_filename}")
                    
                    # Also save full audio
                    self._save_debug_audio(debug_audio_chunks, sample_rate)
                
                # Display latency if manual timing available
                if self.manual_audio_heard_time.value > 0 and self.timestamp_first_audio.value > 0:
                    self._display_pipeline_latency()
                
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
    
    def _display_pipeline_latency(self):
        """Display comprehensive pipeline latency breakdown with manual timing comparison"""
        # Get all timing data
        last_word_frame_to_text = self.last_word_processing_time.value  # Complete frame→text time
        audio_capture_delay = self.audio_capture_delay.value            # Audio capture to processing delay
        stt_internal_time = self.stt_process_time.value                 # STT internal processing time
        stt_total_time = self.stt_total_chunk_time.value                # STT complete function time
        last_text_change = self.last_text_change_time.value             # When STT text last changed
        llm_queue_wait = self.llm_queue_wait_time.value                 # LLM queue waiting time
        llm_to_tts_time = self.llm_to_tts_time.value                    # LLM time-to-TTS
        llm_total_time = self.llm_total_time.value                      # Total LLM processing time
        tts_start = self.timestamp_tts_start.value
        first_audio = self.timestamp_first_audio.value
        
        # Get manual timing data
        manual_speech_end = self.manual_speech_end_time.value
        manual_audio_heard = self.manual_audio_heard_time.value
        
        # Debug: Show all raw timestamps
        print(f"🔍 RAW TIMESTAMPS:")
        print(f"   manual_speech_end: {manual_speech_end:.3f}")
        print(f"   manual_audio_heard: {manual_audio_heard:.3f}")
        print(f"   tts_start: {tts_start:.3f}")
        print(f"   first_audio (chunk ready): {first_audio:.3f}")
        if hasattr(self, 'timestamp_audio_playback_start'):
            print(f"   audio_playback_start: {self.timestamp_audio_playback_start:.3f}")
        else:
            print(f"   audio_playback_start: NOT SET")
        
        # Calculate TTS time-to-first-audio (chunk generation)
        tts_time_to_first_audio = (first_audio - tts_start) * 1000
        
        # Calculate audio chunk to playback delay
        chunk_to_playback_time = 0
        playback_to_heard_time = 0
        if hasattr(self, 'timestamp_audio_playback_start') and self.timestamp_audio_playback_start > 0:
            chunk_to_playback_time = (self.timestamp_audio_playback_start - first_audio) * 1000
            if manual_audio_heard > 0:
                playback_to_heard_time = (manual_audio_heard - self.timestamp_audio_playback_start) * 1000
                print(f"🔍 TIMING CALCULATIONS:")
                print(f"   Chunk to playback: {self.timestamp_audio_playback_start:.3f} - {first_audio:.3f} = {chunk_to_playback_time:.1f}ms")
                print(f"   Playback to heard: {manual_audio_heard:.3f} - {self.timestamp_audio_playback_start:.3f} = {playback_to_heard_time:.1f}ms")
        
        # Calculate new gap measurements
        speech_end_to_stt_complete = 0
        audio_generated_to_heard = 0
        
        if manual_speech_end > 0 and last_text_change > 0:
            speech_end_to_stt_complete = (last_text_change - manual_speech_end) * 1000
        
        if first_audio > 0 and manual_audio_heard > 0:
            audio_generated_to_heard = (manual_audio_heard - first_audio) * 1000
        
        # Calculate comprehensive pipeline times using frame-to-text measurement
        measured_total_frame_based = (last_word_frame_to_text + llm_queue_wait + 
                                     llm_to_tts_time + tts_time_to_first_audio)
        measured_total_component_based = (audio_capture_delay + stt_total_time + llm_queue_wait + 
                                         llm_to_tts_time + tts_time_to_first_audio)
        total_with_complete_llm = (last_word_frame_to_text + llm_queue_wait + 
                                  llm_total_time + tts_time_to_first_audio)
        
        # Calculate manual latency if both events recorded
        manual_latency = 0
        if manual_speech_end > 0 and manual_audio_heard > 0:
            manual_latency = (manual_audio_heard - manual_speech_end) * 1000
        
        # Display results VERY CLEARLY AND LOUDLY
        print("\n🚨🚨🚨 COMPREHENSIVE LATENCY BREAKDOWN 🚨🚨🚨")
        print(f"📊 Last Word Frame→Text: {last_word_frame_to_text:.1f}ms")
        print(f"📊 Audio Capture→Processing: {audio_capture_delay:.1f}ms")
        print(f"📊 STT Internal Processing: {stt_internal_time:.1f}ms")
        print(f"📊 STT Total Function Time: {stt_total_time:.1f}ms")
        print(f"📊 LLM Queue Wait Time: {llm_queue_wait:.1f}ms")
        print(f"🧠 LLM Time-to-TTS: {llm_to_tts_time:.1f}ms")
        print(f"🧠 LLM Total Processing: {llm_total_time:.1f}ms")
        print(f"🎵 TTS Time-to-First-Audio: {tts_time_to_first_audio:.1f}ms")
        print(f"⚡ MEASURED TOTAL (Frame→Audio): {measured_total_frame_based:.1f}ms")
        print(f"📈 MEASURED TOTAL (Components): {measured_total_component_based:.1f}ms")
        print(f"📈 TOTAL WITH COMPLETE LLM: {total_with_complete_llm:.1f}ms")
        
        # New gap measurements and audio timing breakdown
        if speech_end_to_stt_complete > 0:
            print(f"🔍 Speech End → STT Complete: {speech_end_to_stt_complete:.1f}ms")
        
        if chunk_to_playback_time > 0:
            print(f"🔍 Chunk Ready → Audio Playback: {chunk_to_playback_time:.1f}ms")
        
        if playback_to_heard_time > 0:
            print(f"🔍 Audio Playback → Audio Heard: {playback_to_heard_time:.1f}ms")
        
        if audio_generated_to_heard > 0:
            print(f"🔍 Audio Generated → Audio Heard (TOTAL): {audio_generated_to_heard:.1f}ms")
        
        # Manual timing comparison
        if manual_latency > 0:
            print(f"⌨️  MANUAL TOTAL (Speech→Audio): {manual_latency:.1f}ms")
            print(f"🔍 MANUAL vs MEASURED GAP: {manual_latency - measured_total_frame_based:.1f}ms")
            
            # Gap analysis
            if manual_latency > measured_total_frame_based:
                gap = manual_latency - measured_total_frame_based
                print(f"❓ UNACCOUNTED TIME: {gap:.1f}ms")
            else:
                print("✅ Manual timing matches measured components!")
        else:
            print("⌨️  MANUAL TIMING: Not recorded yet")
            print("   Press SPACEBAR when you stop talking, then when you hear audio")
        
        print("🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨\n")
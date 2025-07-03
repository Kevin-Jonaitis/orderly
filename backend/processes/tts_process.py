#!/usr/bin/env python3

import time
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
    
    def run(self):
        """Main TTS process loop - initialize TTS and process text from queue"""
        print("🎵 Starting TTS process...")
        
        # Initialize TTS system with integrated models (like test_tts.py)
        print("🔧 Initializing OrpheusTTS in TTS process...")
        self.tts = OrpheusTTS()
        print("✅ TTS process initialization complete")
        
        # Set up audio streaming (identical to test_tts.py --stream)
        print("🎵 Setting up real-time audio streaming...")
        audio_queue = queue.Queue()
        
        def audio_callback():
            """Stream audio chunks to speakers with low-latency configuration"""
            with sd.OutputStream(samplerate=24000, channels=1, dtype='float32', 
                                blocksize=64, latency='low') as stream:
                first_playback = True
                print(f"🔍 OutputStream info:")
                print(f"   Actual samplerate: {stream.samplerate}")
                print(f"   Actual blocksize: {stream.blocksize}")
                print(f"   Actual latency: {stream.latency}")
                print(f"   Device: {stream.device}")
                
                while True:
                    audio_chunk = audio_queue.get()  # Block until chunk available
                    if audio_chunk is None:  # End signal
                        print("WE FINISHED PULLING AUDIO FROM THE QUEUE ##############################################################")
                        break
                    
                    # Record timestamp when we actually start playing audio
                    if first_playback:
                        timestamp_audio_playback_start = time.time()
                        # Store in a class variable we can access later
                        self.timestamp_audio_playback_start = timestamp_audio_playback_start
                        print(f"🔊 STREAM.WRITE() CALLED: {timestamp_audio_playback_start:.3f}")
                        first_playback = False
                    
                    # Time the actual stream.write() call
                    write_start = time.time()
                    stream.write(audio_chunk.squeeze())
                    write_end = time.time()
                    write_duration = (write_end - write_start) * 1000
                    
                    print(f"🔊 stream.write() took {write_duration:.1f}ms")
        
        # Start audio streaming thread
        audio_thread = threading.Thread(target=audio_callback)
        audio_thread.daemon = True  # Die when main process dies
        audio_thread.start()
        print("🎵 Audio streaming thread started")
        
        # Main loop: wait for text from LLM and generate TTS
        print("🎧 TTS process ready - waiting for text from LLM...")
        while True:
            # Block until we receive text from LLM process
            text = self.tts_text_queue.get()
            
            # Check for shutdown signal
            if text is None:
                print("🛑 TTS process received shutdown signal")
                break
            
            # Record timestamp when TTS receives text from LLM
            tts_receive_time = time.time()
            self.timestamp_tts_start.value = tts_receive_time
            print(f"📥 TTS RECEIVED TEXT: {tts_receive_time:.3f}")
            
            print(f"🎤 TTS received text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Generate TTS audio (text is already stripped from STT/LLM)
            first_chunk = True
            tts_generation_start = time.time()
            print(f"🎵 TTS GENERATION STARTED: {tts_generation_start:.3f}")
            
            # Debug: Collect all audio chunks for analysis
            debug_audio_chunks = []
            
            for result in self.tts.tts_streaming(text, "tara", save_file=None):
                sample_rate, audio_array, chunk_count = result
                
                # Debug: Collect all chunks for analysis
                debug_audio_chunks.append(audio_array.copy())
                
                # Record timestamp for first audio chunk
                if first_chunk:
                    chunk_ready_time = time.time()
                    self.timestamp_first_audio.value = chunk_ready_time
                    print(f"🎵 FIRST CHUNK READY: {chunk_ready_time:.3f}")
                    
                    # Debug: Analyze first chunk for silence
                    chunk_max = np.max(np.abs(audio_array))
                    chunk_rms = np.sqrt(np.mean(audio_array**2))
                    print(f"🔍 First chunk - Max amplitude: {chunk_max:.4f}, RMS: {chunk_rms:.4f}")
                    
                    # Check for leading silence in first chunk
                    silence_threshold = 0.001
                    non_silent_start = 0
                    for i, sample in enumerate(audio_array.flatten()):
                        if abs(sample) > silence_threshold:
                            non_silent_start = i
                            break
                    
                    silent_duration_ms = (non_silent_start / sample_rate) * 1000
                    print(f"🔍 Leading silence in first chunk: {silent_duration_ms:.1f}ms ({non_silent_start} samples)")
                    
                    first_chunk = False
                    
                    # Debug: Time from TTS start to first chunk ready
                    chunk_generation_time = (chunk_ready_time - tts_generation_start) * 1000
                    print(f"🎵 First chunk ready after {chunk_generation_time:.1f}ms")
                    
                    # Time the queue operation
                    queue_start = time.time()
                    audio_queue.put(audio_array)
                    queue_end = time.time()
                    queue_time = (queue_end - queue_start) * 1000
                    print(f"🎵 AUDIO QUEUED: {queue_end:.3f} (took {queue_time:.1f}ms)")
                    
                    # Only display pipeline latency breakdown if manual timing is complete
                    if self.manual_audio_heard_time.value > 0:
                        self._display_pipeline_latency()
                    else:
                        print("🎵 First audio chunk generated - waiting for manual timing...")
                else:
                    # Stream to speakers (subsequent chunks)
                    audio_queue.put(audio_array)
            
            # Debug: Save complete audio for analysis
            if debug_audio_chunks:
                try:
                    # Ensure all chunks are the same shape and properly concatenate
                    print(f"🔍 Concatenating {len(debug_audio_chunks)} audio chunks...")
                    for i, chunk in enumerate(debug_audio_chunks):
                        print(f"   Chunk {i}: shape={chunk.shape}, dtype={chunk.dtype}")
                    
                    # Flatten each chunk and concatenate
                    flattened_chunks = [chunk.flatten() for chunk in debug_audio_chunks]
                    complete_audio = np.concatenate(flattened_chunks, axis=0)
                    
                    # Ensure it's the right shape for mono audio
                    if len(complete_audio.shape) > 1:
                        complete_audio = complete_audio.flatten()
                    
                    debug_filename = f"debug_tts_audio_{int(time.time())}.wav"
                    print(f"🔍 Saving audio: shape={complete_audio.shape}, dtype={complete_audio.dtype}, sr={sample_rate}")
                    
                    # Try multiple save methods
                    try:
                        # Method 1: soundfile with explicit parameters
                        sf.write(debug_filename, complete_audio, sample_rate, format='WAV', subtype='PCM_16')
                        print(f"🔍 Complete TTS audio saved to: {debug_filename}")
                    except Exception as sf_error:
                        print(f"❌ soundfile failed: {sf_error}")
                        try:
                            # Method 2: scipy.io.wavfile as backup
                            from scipy.io import wavfile
                            # Convert float32 to int16 for WAV compatibility
                            audio_int16 = (complete_audio * 32767).astype(np.int16)
                            alt_filename = f"debug_tts_audio_scipy_{int(time.time())}.wav"
                            wavfile.write(alt_filename, sample_rate, audio_int16)
                            print(f"🔍 Complete TTS audio saved to: {alt_filename} (using scipy)")
                        except Exception as scipy_error:
                            print(f"❌ scipy.io.wavfile also failed: {scipy_error}")
                            # Method 3: Save as raw numpy array
                            raw_filename = f"debug_tts_audio_raw_{int(time.time())}.npy"
                            np.save(raw_filename, complete_audio)
                            print(f"🔍 Raw audio saved to: {raw_filename}")
                            print(f"   To convert to WAV manually: sample_rate={sample_rate}, dtype={complete_audio.dtype}")
                    
                    # Analyze complete audio for silence patterns
                    total_samples = len(complete_audio)
                    total_duration_ms = (total_samples / sample_rate) * 1000
                    
                    # Find first non-silent sample in complete audio
                    silence_threshold = 0.001
                    first_audio_sample = 0
                    for i, sample in enumerate(complete_audio):
                        if abs(sample) > silence_threshold:
                            first_audio_sample = i
                            break
                    
                    leading_silence_ms = (first_audio_sample / sample_rate) * 1000
                    print(f"🔍 COMPLETE AUDIO ANALYSIS:")
                    print(f"   Total duration: {total_duration_ms:.1f}ms ({total_samples} samples)")
                    print(f"   Leading silence: {leading_silence_ms:.1f}ms ({first_audio_sample} samples)")
                    print(f"   Max amplitude: {np.max(np.abs(complete_audio)):.4f}")
                    print(f"   RMS: {np.sqrt(np.mean(complete_audio**2)):.4f}")
                    
                except Exception as e:
                    print(f"❌ Error saving/analyzing audio: {e}")
                    # Save raw data for manual inspection
                    np.save(f"debug_tts_raw_{int(time.time())}.npy", debug_audio_chunks)
                    print(f"🔍 Raw audio chunks saved as .npy file for manual analysis")
            
            # Signal end of this TTS generation
            print("✅ TTS generation complete")
            
            # Check if manual timing became available during TTS generation
            if self.manual_audio_heard_time.value > 0 and self.timestamp_first_audio.value > 0:
                self._display_pipeline_latency()
        
        # Clean up audio streaming
        print("🔧 Cleaning up TTS audio streaming...")
        audio_queue.put(None)  # End signal
        time.sleep(0.5)  # Let the audio finish playing
        audio_thread.join()  # Wait for audio to finish
        print("🛑 TTS process shutdown complete")
    
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
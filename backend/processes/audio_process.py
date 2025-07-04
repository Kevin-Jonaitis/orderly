#!/usr/bin/env python3

import time as time_module
import multiprocessing
from multiprocessing import Process
import pyaudio
import numpy as np
import pickle
import queue
import os


class AudioProcessor(Process):
    """Dedicated audio processing - handles all audio streaming operations
    
    This process eliminates the threading complexity from TTSProcess and provides
    a clean separation of concerns. Audio handling is modeled after the working
    test_tts_audio_minimal.py structure for optimal performance.
    """
    
    def __init__(self, audio_queue, first_audio_chunk_timestamp, use_blocking_audio=True):
        super().__init__(name="AudioProcess")
        self.audio_queue = audio_queue  # multiprocessing.Queue
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.use_blocking_audio = use_blocking_audio  # PyAudio works best with blocking
        self.audio_stream = None
        self.pyaudio_instance = None
        self.first_playback = True
        
    # Debug chunk loading removed - handled externally by test scripts or TTSProcess
    
    def run(self):
        """Main audio process - pure audio playback"""
        print("🎵 [AudioProcessor] Starting audio processing...")
        
        # Handle audio with minimal complexity
        self._handle_audio()
        
    def _handle_audio(self):
        """Direct audio handling with PyAudio"""
        print(f"🎵 [AudioProcessor] Starting PyAudio handling (mode: {'blocking' if self.use_blocking_audio else 'callback'})")
        
        if self.use_blocking_audio:
            self._pyaudio_blocking()
        else:
            self._pyaudio_callback()
    
    def _pyaudio_blocking(self):
        """PyAudio blocking implementation for low latency"""
        try:
            print("🎵 [AudioProcessor] Using PyAudio blocking mode")
            
            # Initialize PyAudio with WSL2 compatibility
            os.environ['ALSA_PCM_CARD'] = 'default'
            os.environ['ALSA_PCM_DEVICE'] = '0'
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find working audio device
            working_device = self._find_working_device()
            if working_device is None:
                print("❌ [AudioProcessor] No working audio devices found")
                return
            
            # Create PyAudio stream with minimal latency
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=24000,
                output=True,
                frames_per_buffer=512,  # Much smaller than sounddevice's 2048
                output_device_index=working_device,
                start=False
            )
            
            print(f"🔊 [AudioProcessor] PyAudio config: 512 samples ({512/24000*1000:.1f}ms buffer)")
            
            # Start stream
            self.audio_stream.start_stream()
            print("▶️  [AudioProcessor] PyAudio stream started...")
            
            # Simple blocking write loop
            while True:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Record first write timing
                    if self.first_playback:
                        print(f"🔊 [AudioProcessor] FIRST WRITE: {time_module.time():.3f}")
                        self.first_playback = False
                    
                    # Convert to bytes for PyAudio
                    audio_chunk = audio_chunk.flatten().astype(np.float32)
                    audio_bytes = audio_chunk.tobytes()
                    
                    # Direct write to PyAudio
                    self.audio_stream.write(audio_bytes)
                    
                except queue.Empty:
                    # No audio available - continue waiting
                    continue
                except Exception as e:
                    print(f"❌ [AudioProcessor] Error in PyAudio write: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ [AudioProcessor] Error in PyAudio blocking: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_pyaudio()
    
    def _pyaudio_callback(self):
        """PyAudio callback implementation (not recommended for WSL2)"""
        print("⚠️  [AudioProcessor] PyAudio callback mode not implemented - using blocking mode instead")
        self._pyaudio_blocking()
    
    def _find_working_device(self):
        """Find a working audio device in WSL2 environment"""
        print("🔍 [AudioProcessor] Finding working audio device...")
        
        for i in range(self.pyaudio_instance.get_device_count()):
            try:
                info = self.pyaudio_instance.get_device_info_by_index(i)
                if info['maxOutputChannels'] > 0:
                    # Try to open a test stream
                    test_stream = self.pyaudio_instance.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=24000,
                        output=True,
                        frames_per_buffer=512,
                        output_device_index=i,
                        start=False
                    )
                    test_stream.close()
                    print(f"✅ [AudioProcessor] Found working device: [{i}] {info['name']}")
                    return i
            except Exception as e:
                continue
        
        print("❌ [AudioProcessor] No working audio devices found")
        return None
    
    def _cleanup_pyaudio(self):
        """Clean up PyAudio resources"""
        try:
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            print("✅ [AudioProcessor] PyAudio cleanup complete")
        except Exception as e:
            print(f"❌ [AudioProcessor] Cleanup error: {e}")
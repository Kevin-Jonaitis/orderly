#!/usr/bin/env python3

import time as time_module
import multiprocessing
from multiprocessing import Process
import numpy as np
import queue


class AudioProcessor(Process):
    """Dedicated audio processing - handles all audio streaming operations
    
    This process eliminates the threading complexity from TTSProcess and provides
    a clean separation of concerns. Audio handling is modeled after the working
    test_tts_audio_minimal.py structure for optimal performance.
    """
    
    def __init__(self, audio_queue, first_audio_chunk_timestamp, audio_output_webrtc_queue=None):
        super().__init__(name="AudioProcess")
        self.audio_queue = audio_queue  # multiprocessing.Queue
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.audio_output_webrtc_queue = audio_output_webrtc_queue  # Queue for WebRTC audio output
        
        # Audio processing parameters for WebRTC
        self.tts_sample_rate = 24000  # TTS output rate
        self.webrtc_sample_rate = 48000  # WebRTC expected rate
        self.audio_buffer = np.array([], dtype=np.float32)
        
    # Debug chunk loading removed - handled externally by test scripts or TTSProcess
    
    def run(self):
        """Main audio process - WebRTC streaming only"""
        print("🎵 [AudioProcessor] Starting WebRTC audio processing...")
        
        # Process audio for WebRTC streaming only
        self._process_webrtc_audio()
    
    def _process_webrtc_audio(self):
        """Process audio for WebRTC streaming only"""
        print("🎵 [AudioProcessor] Starting WebRTC audio processing loop...")
        
        try:
            while True:
                try:
                    # Get audio chunk from TTS
                    audio_chunk = self.audio_queue.get(timeout=1.0)
                    
                    # Record first audio timing
                    if not hasattr(self, '_first_audio_received'):
                        print(f"🎵 [AudioProcessor] FIRST AUDIO: {time_module.time():.3f}")
                        self._first_audio_received = True
                    
                    # Process audio for WebRTC streaming
                    if self.audio_output_webrtc_queue is not None:
                        self._process_audio_for_webrtc(audio_chunk)
                    
                except queue.Empty:
                    # No audio available - continue waiting
                    continue
                except Exception as e:
                    print(f"❌ [AudioProcessor] Error in WebRTC audio processing: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ [AudioProcessor] Error in WebRTC audio loop: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_audio_for_webrtc(self, audio_chunk):
        """Process audio chunk for WebRTC streaming"""
        try:
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            
            # Process when we have enough audio for a WebRTC frame
            webrtc_frame_samples = int(self.webrtc_sample_rate * 0.02)  # 20ms frame
            
            while len(self.audio_buffer) >= webrtc_frame_samples:
                # Extract frame
                frame_audio = self.audio_buffer[:webrtc_frame_samples]
                self.audio_buffer = self.audio_buffer[webrtc_frame_samples:]
                
                # Resample from 24kHz to 48kHz
                import scipy.signal
                frame_audio_48k = scipy.signal.resample(frame_audio, 
                                                      int(len(frame_audio) * self.webrtc_sample_rate / self.tts_sample_rate))
                
                # Convert to int16 for WebRTC
                frame_audio_int16 = (frame_audio_48k * 32767).astype(np.int16)
                
                # Send to WebRTC queue
                try:
                    self.audio_output_webrtc_queue.put_nowait(frame_audio_int16)
                except queue.Full:
                    # Queue is full, skip this frame
                    print("⚠️ [AudioProcessor] WebRTC queue full, skipping frame")
                    
        except Exception as e:
            print(f"❌ [AudioProcessor] Error processing audio for WebRTC: {e}")
    

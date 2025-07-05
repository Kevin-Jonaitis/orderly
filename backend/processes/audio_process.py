#!/usr/bin/env python3

import time as time_module
import multiprocessing
from multiprocessing import Process
import numpy as np
import queue
import scipy.signal


class AudioProcessor(Process):
    """Dedicated audio processing - handles all audio streaming operations
    
    This process eliminates the threading complexity from TTSProcess and provides
    a clean separation of concerns. Audio handling is modeled after the working
    test_tts_audio_minimal.py structure for optimal performance.
    """
    
    def __init__(self, audio_queue, first_audio_chunk_timestamp, audio_output_webrtc_queue):
        super().__init__(name="AudioProcess")
        self.audio_queue = audio_queue  # multiprocessing.Queue
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.audio_output_webrtc_queue = audio_output_webrtc_queue  # Queue for WebRTC audio output
        
        # Audio processing parameters for WebRTC
        self.tts_sample_rate = 24000  # TTS output rate
        self.webrtc_sample_rate = 48000  # WebRTC expected rate
        self.webrtc_frame_duration = 0.02  # 20ms frames
        self.webrtc_samples_per_frame = int(self.webrtc_sample_rate * self.webrtc_frame_duration)  # 960 samples
        self.tts_samples_per_frame = int(self.tts_sample_rate * self.webrtc_frame_duration)  # 480 samples at 24kHz
        
        # Audio buffer for frame processing
        self.audio_buffer = np.array([], dtype=np.float32)  # Buffer to accumulate audio chunks
        
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
                    # Get audio chunk from TTS (blocking)
                    audio_chunk = self.audio_queue.get()
                    
                    # Record first audio timing
                    if not hasattr(self, '_first_audio_received'):
                        print(f"🎵 [AudioProcessor] FIRST AUDIO: {time_module.time():.3f}")
                        self._first_audio_received = True
                    
                    # Process audio for WebRTC streaming
                    self._process_audio_for_webrtc(audio_chunk)
                    
                except Exception as e:
                    print(f"❌ [AudioProcessor] Error in WebRTC audio processing: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ [AudioProcessor] Error in WebRTC audio loop: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_audio_for_webrtc(self, audio_chunk):
        """Process audio chunk for WebRTC streaming using nearest neighbor upsampling"""
        try:
            # Ensure audio_chunk is 1D (mono)
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            print(f"🎵 [AudioProcessor] Received chunk: {len(audio_chunk)} samples, buffer: {len(self.audio_buffer)} samples")
            
            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
            
            # Process complete frames from buffer
            frames_sent = 0
            while len(self.audio_buffer) >= self.tts_samples_per_frame:
                # Extract exactly one frame worth of audio (480 samples at 24kHz = 20ms)
                frame_audio = self.audio_buffer[:self.tts_samples_per_frame]
                self.audio_buffer = self.audio_buffer[self.tts_samples_per_frame:]
                
                # Upsample from 24kHz to 48kHz using nearest neighbor (simple duplication)
                frame_audio_48k = np.repeat(frame_audio, 2)
                
                # Convert to int16
                frame_int16 = (frame_audio_48k * 32767).astype(np.int16)
                
                print(f"🎵 [AudioProcessor] Sending frame: {len(frame_int16)} samples, max: {np.max(np.abs(frame_int16))}")
                
                # Send single frame to WebRTC queue
                try:
                    self.audio_output_webrtc_queue.put_nowait(frame_int16)
                    frames_sent += 1
                except queue.Full:
                    print("⚠️ [AudioProcessor] WebRTC queue full, skipping frame")
            
            if frames_sent > 0:
                print(f"🎵 [AudioProcessor] Sent {frames_sent} frames, buffer remaining: {len(self.audio_buffer)} samples")
                
        except Exception as e:
            print(f"❌ [AudioProcessor] Error processing audio for WebRTC: {e}")
            import traceback
            traceback.print_exc()
    

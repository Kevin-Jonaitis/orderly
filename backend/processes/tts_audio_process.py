#!/usr/bin/env python3

import time as time_module
import multiprocessing
from multiprocessing import Process
import numpy as np
import queue
import scipy.signal
import time
from typing import Optional


class AudioProcessor(Process):
    """
    Process TTS audio chunks for WebRTC and WebSocket streaming with ultra-low latency optimizations
    """
    
    def __init__(self, audio_queue, first_audio_chunk_timestamp, audio_output_webrtc_queue, audio_output_websocket_queue=None):
        super().__init__(name="AudioProcess")
        self.audio_queue = audio_queue
        self.first_audio_chunk_timestamp = first_audio_chunk_timestamp
        self.audio_output_webrtc_queue = audio_output_webrtc_queue
        self.audio_output_websocket_queue = audio_output_websocket_queue  # Queue for WebSocket audio output
        
        # Audio configuration for ultra-low latency
        self.tts_sample_rate = 24000  # TTS output rate
        self.webrtc_sample_rate = 48000  # WebRTC expected rate
        self.webrtc_frame_duration = 0.02  # 20ms frames for WebRTC
        self.webrtc_samples_per_frame = int(self.webrtc_sample_rate * self.webrtc_frame_duration)  # 960 samples
        self.tts_samples_per_frame = int(self.tts_sample_rate * self.webrtc_frame_duration)  # 480 samples at 24kHz
        
        # Audio buffer for frame processing
        self.audio_buffer = np.array([], dtype=np.float32)  # Buffer to accumulate audio chunks
        
        # Timing tracking
        self.first_chunk_read_time = None
        self.first_webrtc_output_time = None
        self.first_websocket_output_time = None
        self.chunk_count = 0
        
        print(f"🎵 [AudioProcessor] Initialized with ultra-low latency settings")
        print(f" [AudioProcessor] TTS rate: {self.tts_sample_rate}Hz, WebRTC rate: {self.webrtc_sample_rate}Hz")
        print(f"🎵 [AudioProcessor] Frame duration: {self.webrtc_frame_duration*1000:.1f}ms")
        print(f" [AudioProcessor] TTS samples per frame: {self.tts_samples_per_frame}")
        print(f"🎵 [AudioProcessor] WebRTC samples per frame: {self.webrtc_samples_per_frame}")

    def run(self):
        """Main processing loop with ultra-low latency optimizations"""
        print("🎵 [AudioProcessor] Starting audio processing loop...")
        
        while True:
            try:
                # Get audio chunk from TTS (blocking)
                chunk_read_start = time.time()
                audio_chunk = self.audio_queue.get()
                chunk_read_end = time.time()
                
                # Track first chunk timing
                if self.first_chunk_read_time is None:
                    self.first_chunk_read_time = chunk_read_end
                    print(f"⏱️ [AudioProcessor] First chunk read at: {self.first_chunk_read_time:.6f}s")
                
                # Check for termination signal
                if audio_chunk is None:
                    print("🎵 [AudioProcessor] Received termination signal")
                    break
                
                # Process audio for both WebRTC and WebSocket
                self._process_audio_for_webrtc(audio_chunk, chunk_read_end)
                
            except Exception as e:
                print(f"❌ [AudioProcessor] Error in processing loop: {e}")
                break
        
        print("🎵 [AudioProcessor] Audio processing loop ended")

    def _process_audio_for_webrtc(self, audio_chunk, chunk_read_time):
        """Process audio chunk for WebRTC streaming using nearest neighbor upsampling with ultra-low latency"""
        
        # Ensure audio_chunk is 1D (mono)
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()
        
        self.chunk_count += 1
        # print(f"🎵 [AudioProcessor] Processing chunk #{self.chunk_count}: {len(audio_chunk)} samples, buffer: {len(self.audio_buffer)} samples")
        
        # Add to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
        # Process complete frames from buffer
        frames_sent = 0
        
        while len(self.audio_buffer) >= self.tts_samples_per_frame:
            # Extract frame
            frame_audio = self.audio_buffer[:self.tts_samples_per_frame]
            self.audio_buffer = self.audio_buffer[self.tts_samples_per_frame:]
            
            # Upsample from 24kHz to 48kHz using nearest neighbor for ultra-low latency
            # This is faster than interpolation and reduces latency
            upsampled_frame = self._nearest_neighbor_upsample(frame_audio, 2)
            
            # Convert to int16 for WebRTC
            frame_int16 = (upsampled_frame * 32767).astype(np.int16)
            
            # Send to WebRTC queue with timing
            if self.audio_output_webrtc_queue is not None:
                try:
                    webrtc_output_time = time.time()
                    self.audio_output_webrtc_queue.put_nowait(frame_int16)
                    frames_sent += 1
                    
                    # Track first WebRTC output timing
                    if self.first_webrtc_output_time is None:
                        self.first_webrtc_output_time = webrtc_output_time
                        webrtc_latency = (self.first_webrtc_output_time - self.first_chunk_read_time) * 1000
                        print(f"⏱️ [AudioProcessor] First WebRTC output at: {self.first_webrtc_output_time:.6f}s")
                        print(f"⏱️ [AudioProcessor] Audio Queue → WebRTC Queue latency: {webrtc_latency:.2f}ms")
                    
                except:
                    print("⚠️ [AudioProcessor] WebRTC queue full, skipping frame")
            
            # Send single frame to WebSocket queue (immediate streaming like RealtimeVoiceChat)
            if self.audio_output_websocket_queue is not None:
                try:
                    websocket_output_time = time.time()
                    self.audio_output_websocket_queue.put_nowait(frame_int16)
                    
                    # Track first WebSocket output timing
                    if self.first_websocket_output_time is None:
                        self.first_websocket_output_time = websocket_output_time
                        websocket_latency = (self.first_websocket_output_time - self.first_chunk_read_time) * 1000
                        print(f"⏱️ [AudioProcessor] First WebSocket output at: {self.first_websocket_output_time:.6f}s")
                        print(f"⏱️ [AudioProcessor] Audio Queue → WebSocket Queue latency: {websocket_latency:.2f}ms")
                    
                except:
                    print("⚠️ [AudioProcessor] WebSocket queue full, skipping frame")
            else:
                print("❌ [AudioProcessor] WebSocket queue is None!")
        
        # if frames_sent > 0:
        #     print(f"🎵 [AudioProcessor] Sent {frames_sent} frames to WebRTC queue")
        #     print(f"🎵 [AudioProcessor] Sent {frames_sent} frames to WebSocket queue")
        #     print(f"🎵 [AudioProcessor] Buffer remaining: {len(self.audio_buffer)} samples")

    def _nearest_neighbor_upsample(self, audio, factor):
        """Ultra-fast nearest neighbor upsampling for minimal latency"""
        # Simple repeat-based upsampling (faster than interpolation)
        upsampled = np.repeat(audio, factor)
        return upsampled 
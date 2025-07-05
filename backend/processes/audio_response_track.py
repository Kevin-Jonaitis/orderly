#!/usr/bin/env python3
"""
Audio Response Track

Handles outgoing WebRTC audio frames to frontend.
Reads pre-processed audio data from queue and sends to frontend.
"""

import multiprocessing
import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame


class AudioResponseTrack(MediaStreamTrack):
    """
    Audio response track - streams processed audio back to frontend
    Reads WebRTC-ready audio frames from queue and sends to frontend
    """
    
    kind = "audio"
    direction = "sendonly"  # This track only sends audio to the frontend
    
    def __init__(self, audio_output_webrtc_queue: multiprocessing.Queue, connection_id: str):
        super().__init__()
        self.audio_output_webrtc_queue = audio_output_webrtc_queue
        self.connection_id = connection_id
        self.frame_count = 0
        self.webrtc_sample_rate = 48000  # WebRTC expected rate
        self.frame_duration = 0.02  # 20ms frames
        self.samples_per_frame = int(self.webrtc_sample_rate * self.frame_duration)
        print(f"🎵 [AudioResponseTrack] Created for {connection_id}")
    
    async def recv(self):
        """Generate WebRTC audio frames from processed audio queue"""
        self.frame_count += 1
        
        # Try to get processed audio from queue
        try:
            # Get audio data from queue (should be WebRTC-ready int16 data)
            audio_data = self.audio_output_webrtc_queue.get_nowait()
            
            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.int16)
            
            # Ensure int16 format for WebRTC
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Take samples for this frame
            if len(audio_data) >= self.samples_per_frame:
                frame_audio = audio_data[:self.samples_per_frame]
                # Put remaining audio back in queue for next frame
                remaining_audio = audio_data[self.samples_per_frame:]
                if len(remaining_audio) > 0:
                    self.audio_output_webrtc_queue.put_nowait(remaining_audio)
            else:
                # Pad with zeros if not enough audio
                frame_audio = np.pad(audio_data, (0, self.samples_per_frame - len(audio_data)), 'constant')
            
            if self.frame_count <= 3 or self.frame_count % 100 == 0:
                print(f"🎵 [AudioResponseTrack] Frame {self.frame_count}: {frame_audio.shape} samples, max: {np.max(np.abs(frame_audio))}")
                
        except Exception as e:
            # Queue is empty or error - generate silence frame
            frame_audio = np.zeros(self.samples_per_frame, dtype=np.int16)
            if self.frame_count % 100 == 0:
                print(f"🎵 [AudioResponseTrack] Frame {self.frame_count}: Silence frame (queue empty)")
        
        # Create WebRTC audio frame
        frame = AudioFrame.from_ndarray(
            frame_audio.reshape(-1, 1),  # Reshape to (samples, channels)
            format="s16",
            layout="mono"
        )
        frame.sample_rate = self.webrtc_sample_rate
        frame.pts = self.frame_count * self.samples_per_frame
        frame.time_base = 1 / self.webrtc_sample_rate
        
        return frame 
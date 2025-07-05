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
import time
import os
import wave
from fractions import Fraction


class AudioResponseTrack(MediaStreamTrack):
    """
    Audio response track - streams processed audio back to frontend
    Reads WebRTC-ready audio frames from queue and sends to frontend
    """    
    kind = "audio"

    def __init__(self, audio_output_webrtc_queue: multiprocessing.Queue, connection_id: str):
        super().__init__()
        self.audio_output_webrtc_queue = audio_output_webrtc_queue
        self.connection_id = connection_id
        self.frame_count = 0
        self.webrtc_sample_rate = 48000  # WebRTC expected rate
        self.frame_duration = 0.02  # 20ms frames
        self.samples_per_frame = int(self.webrtc_sample_rate * self.frame_duration)
        
        # Frame debugging stats
        self.frame_stats = {
            'sent_frames': 0,
            'silence_frames': 0,
            'queue_empty_count': 0,
            'last_frame_time': 0,
            'frame_intervals': [],
            'audio_levels': []
        }
        
        # Audio recording for debugging
        self.recorded_audio = []
        self.last_save_time = time.time()
        self.save_interval = 5.0  # Save every 5 seconds
        self.recording_dir = "audio_debug"
        
        # Create recording directory
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir)
        
        print(f"🎵 [AudioResponseTrack] Created for {connection_id}")
        print(f"🎵 [AudioResponseTrack] Will save audio recordings to {self.recording_dir}/")
    
    async def recv(self):
        """Generate WebRTC audio frames from processed audio queue"""
        
        try:
            frame_start_time = time.time()
            self.frame_count += 1
            self.frame_stats['sent_frames'] += 1
            
            # Track frame timing
            current_time = time.time()
            if self.frame_stats['last_frame_time'] > 0:
                interval = (current_time - self.frame_stats['last_frame_time']) * 1000  # Convert to ms
                self.frame_stats['frame_intervals'].append(interval)
                
                # Keep only last 100 intervals
                if len(self.frame_stats['frame_intervals']) > 100:
                    self.frame_stats['frame_intervals'].pop(0)
                
                # Detect frame timing issues (intervals > 30ms are suspicious for 20ms frames)
                if interval > 30:
                    print(f"⚠️ [AudioResponseTrack] Frame timing issue: {interval:.1f}ms interval (expected ~20ms)")
            
            self.frame_stats['last_frame_time'] = current_time
            
            # Try to get processed audio from queue
            try:
                # Check queue size
                queue_size = self.audio_output_webrtc_queue.qsize()
                if queue_size == 0:
                    self.frame_stats['queue_empty_count'] += 1
                    if self.frame_stats['queue_empty_count'] % 10 == 0:  # Log every 10th empty queue
                        print(f"⚠️ [AudioResponseTrack] Queue empty {self.frame_stats['queue_empty_count']} times")
                
                # Get audio data from queue (should be WebRTC-ready int16 data)
                audio_data = self.audio_output_webrtc_queue.get_nowait()
                
                # Track audio levels
                audio_max = np.max(np.abs(audio_data))
                self.frame_stats['audio_levels'].append(audio_max)
                if len(self.frame_stats['audio_levels']) > 100:
                    self.frame_stats['audio_levels'].pop(0)
                
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
                
                # Record audio for debugging
                self.recorded_audio.extend(frame_audio.tolist())
                
                # Check if it's time to save the recording
                current_time = time.time()
                if current_time - self.last_save_time >= self.save_interval:
                    self._save_audio_recording()
                    self.last_save_time = current_time
                
                if self.frame_count <= 3 or self.frame_count % 100 == 0:
                    avg_interval = sum(self.frame_stats['frame_intervals']) / len(self.frame_stats['frame_intervals']) if self.frame_stats['frame_intervals'] else 0
                    avg_level = sum(self.frame_stats['audio_levels']) / len(self.frame_stats['audio_levels']) if self.frame_stats['audio_levels'] else 0
                    print(f"🎵 [AudioResponseTrack] Frame {self.frame_count}: {frame_audio.shape} samples, max: {np.max(np.abs(frame_audio))}, avg_interval: {avg_interval:.1f}ms, avg_level: {avg_level:.1f}")
                    
            except Exception as e:
                # Queue is empty or error - generate silence frame
                self.frame_stats['silence_frames'] += 1
                frame_audio = np.zeros(self.samples_per_frame, dtype=np.int16)
                if self.frame_count % 100 == 0:
                    print(f"🎵 [AudioResponseTrack] Frame {self.frame_count}: Silence frame (queue empty) - Total silence: {self.frame_stats['silence_frames']}")
            
            # Create WebRTC audio frame
            frame = AudioFrame(format="s16", layout="mono", samples=len(frame_audio))
            # Update the frame with our audio data
            frame.planes[0].update(frame_audio.tobytes())
            frame.sample_rate = self.webrtc_sample_rate
            frame.pts = self.frame_count * self.samples_per_frame
            frame.time_base = Fraction(1, self.webrtc_sample_rate)

            return frame
            
        except Exception as e:
            print(f"❌ [AudioResponseTrack] Error in recv(): {e}")
            import traceback
            traceback.print_exc()
            # Re-raise the exception so the WebRTC stack knows something went wrong
            raise
    
    def _save_audio_recording(self):
        """Save recorded audio to a WAV file"""
        if not self.recorded_audio:
            print(f"🎵 [AudioResponseTrack] No audio to save")
            return
        
        try:
            # Convert to numpy array
            audio_array = np.array(self.recorded_audio, dtype=np.int16)
            
            # Create filename with timestamp
            timestamp = int(time.time())
            filename = f"{self.recording_dir}/audio_response_{self.connection_id}_{timestamp}.wav"
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.webrtc_sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            
            print(f"🎵 [AudioResponseTrack] Saved {len(audio_array)} samples to {filename}")
            print(f"🎵 [AudioResponseTrack] Audio duration: {len(audio_array) / self.webrtc_sample_rate:.2f} seconds")
            print(f"🎵 [AudioResponseTrack] Max amplitude: {np.max(np.abs(audio_array))}")
            
            # Clear recorded audio for next recording
            self.recorded_audio = []
            
        except Exception as e:
            print(f"❌ [AudioResponseTrack] Error saving audio: {e}")
            # Clear recorded audio to prevent memory buildup
            self.recorded_audio = [] 
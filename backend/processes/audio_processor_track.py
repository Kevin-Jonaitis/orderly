#!/usr/bin/env python3
"""
Audio Processor Track

Handles incoming WebRTC audio frames and processes them for STT.
Moved from webrtc.py to its own file for better organization.
"""

import asyncio
import collections
import multiprocessing
import numpy as np
import soundfile as sf
import scipy.signal
from aiortc import MediaStreamTrack
from av import AudioFrame
import time


class AudioProcessorTrack(MediaStreamTrack):
    """
    Audio track processor - processes incoming audio frames and queues them for STT
    """
    
    kind = "audio"
    
    def __init__(self, track: MediaStreamTrack, audio_queue: multiprocessing.Queue, connection_id: str):
        super().__init__()
        self.track = track
        self.audio_queue = audio_queue
        self.connection_id = connection_id
        self.frame_count = 0
        self.sample_rate = None
        self.rolling_buffer = collections.deque()
        self.rolling_buffer_samples = 0
        self.max_seconds = 5
        self.max_samples = None  # Will be set after first frame
        
        # Frame debugging stats
        self.frame_stats = {
            'received_frames': 0,
            'processed_frames': 0,
            'dropped_frames': 0,
            'queue_full_count': 0,
            'last_frame_time': 0,
            'frame_intervals': [],
            'processing_times': []
        }
        
        print(f"🎧 [WebRTC] AudioProcessorTrack created for {connection_id}")
    
    async def recv(self):
        """Process audio frames from browser"""
        frame_start_time = time.time()
        frame = await self.track.recv()
        self.frame_count += 1
        self.frame_stats['received_frames'] += 1
        
        # Track frame timing
        current_time = time.time()
        if self.frame_stats['last_frame_time'] > 0:
            interval = (current_time - self.frame_stats['last_frame_time']) * 1000  # Convert to ms
            self.frame_stats['frame_intervals'].append(interval)
            
            # Keep only last 100 intervals
            if len(self.frame_stats['frame_intervals']) > 100:
                self.frame_stats['frame_intervals'].pop(0)
            
            # Detect frame drops (intervals > 50ms are suspicious)
            if interval > 50:
                self.frame_stats['dropped_frames'] += 1
                print(f"⚠️ [WebRTC] Potential frame drop detected: {interval:.1f}ms interval")
        
        self.frame_stats['last_frame_time'] = current_time
        
        # Convert to numpy array (mono, int16, 48kHz)
        audio_array = frame.to_ndarray().flatten()
        
        # Remove duplication: take every other sample (if needed)
        audio_array = audio_array[::2]  # Commented out to test if this causes buzzing
        
        # Now audio_array is mono, int16, 48kHz
        # Convert to float32 in range [-1, 1] for STT processing
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        # Resample from 48kHz to 16kHz for conformer_stream_step
        # Use decimate for cleaner downsampling (less artifacts than resample)
        audio_array_16k = scipy.signal.decimate(audio_array, 3, n=8)
        
        # Set sample rate and max_samples on first frame
        if self.sample_rate is None:
            # Conformer expects 16kHz mono float32
            self.sample_rate = 16000
            self.max_samples = int(self.max_seconds * self.sample_rate)
        
        # Rolling buffer logic (only if max_samples is set)
        if self.max_samples is not None:
            self.rolling_buffer.append(audio_array_16k)
            self.rolling_buffer_samples += len(audio_array_16k)
            # Trim buffer to last 5 seconds
            while self.rolling_buffer_samples > self.max_samples:
                removed = self.rolling_buffer.popleft()
                self.rolling_buffer_samples -= len(removed)
        
        # Queue for STT processing (send 16kHz float32 mono)
        try:
            # Check queue size before putting
            queue_size = self.audio_queue.qsize()
            if queue_size > 50:  # Queue getting full
                self.frame_stats['queue_full_count'] += 1
				# THIS IS GOOD STATS, DO NOT REMOVE
                print(f"⚠️ [WebRTC] Queue getting full: {queue_size} items")
            
            self.audio_queue.put_nowait(audio_array_16k)
            self.frame_stats['processed_frames'] += 1
            
            # Track processing time
            processing_time = (time.time() - frame_start_time) * 1000  # Convert to ms
            self.frame_stats['processing_times'].append(processing_time)
            
            # Keep only last 100 processing times
            if len(self.frame_stats['processing_times']) > 100:
                self.frame_stats['processing_times'].pop(0)
            
            if self.frame_count <= 3:
                print(f"🎤 [WebRTC] Frame {self.frame_count}: {audio_array_16k.shape} samples, max: {np.max(np.abs(audio_array_16k)):.4f}, proc_time: {processing_time:.1f}ms")
            elif self.frame_count % 50 == 0:
                avg_processing = sum(self.frame_stats['processing_times']) / len(self.frame_stats['processing_times'])
                avg_interval = sum(self.frame_stats['frame_intervals']) / len(self.frame_stats['frame_intervals']) if self.frame_stats['frame_intervals'] else 0
                print(f"🎤 [WebRTC] Processed {self.frame_count} frames - Avg proc: {avg_processing:.1f}ms, Avg interval: {avg_interval:.1f}ms, Dropped: {self.frame_stats['dropped_frames']}")
                
        except Exception as e:
            print(f"❌ [WebRTC] Queue error: {e}")
            self.frame_stats['dropped_frames'] += 1
        
        # Save last 5 seconds to file every 200 frames (for testing)
        if self.frame_count % 200 == 0:
            self.save_last_5_seconds_to_wav(f"audio_recording_{self.connection_id}_{self.frame_count}.wav")
        
        return frame
    
    def save_last_5_seconds_to_wav(self, filename):
        """Save the last 5 seconds of audio to a WAV file (16kHz mono float32)."""
        if self.sample_rate is None or self.rolling_buffer_samples == 0:
            print(f"[WebRTC] No audio to save.")
            return
        
        # Use rolling buffer audio (float32, 16kHz)
        audio_data = np.concatenate(list(self.rolling_buffer))
        
        # Only keep the last 5 seconds worth of 16kHz audio
        if self.max_samples is not None and len(audio_data) > self.max_samples:
            audio_data = audio_data[-self.max_samples:]
        
        print(f"[DEBUG] Saving WAV: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min()}, max={audio_data.max()}, sample_rate={self.sample_rate}")
        try:
            sf.write(filename, audio_data, self.sample_rate, subtype='FLOAT')
            print(f"💾 [WebRTC] Saved last 5 seconds to {filename} ({len(audio_data)} samples @ {self.sample_rate} Hz)")
        except Exception as e:
            print(f"❌ [WebRTC] Failed to save audio: {e}") 
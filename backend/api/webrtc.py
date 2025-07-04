"""
WebRTC server - direct port of aiortc server.py example for FastAPI
Adapted for audio-only streaming to STT processor
"""

import asyncio
import json
import multiprocessing
import time
from typing import Optional

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from fastapi import FastAPI, HTTPException
import collections
import soundfile as sf
import scipy.signal

# Global state - exact copy from aiortc server.py
pcs = set()


class AudioProcessorTrack(MediaStreamTrack):
    """
    Audio track processor - replaces VideoTransformTrack from aiortc example
    Processes incoming audio frames and queues them for STT
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
        print(f"🎧 [WebRTC] AudioProcessorTrack created for {connection_id}")
    
    async def recv(self):
        """Process audio frames from browser"""
        frame = await self.track.recv()
        self.frame_count += 1
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
            self.audio_queue.put_nowait(audio_array_16k)
            if self.frame_count <= 3:
                print(f"🎤 [WebRTC] Frame {self.frame_count}: {audio_array_16k.shape} samples, max: {np.max(np.abs(audio_array_16k)):.4f}")
            elif self.frame_count % 50 == 0:
                print(f"🎤 [WebRTC] Processed {self.frame_count} frames")
        except Exception as e:
            print(f"❌ [WebRTC] Queue error: {e}")
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
        if len(audio_data) > self.max_samples:
            audio_data = audio_data[-self.max_samples:]
        
        print(f"[DEBUG] Saving WAV: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min()}, max={audio_data.max()}, sample_rate={self.sample_rate}")
        try:
            sf.write(filename, audio_data, self.sample_rate, subtype='FLOAT')
            print(f"💾 [WebRTC] Saved last 5 seconds to {filename} ({len(audio_data)} samples @ {self.sample_rate} Hz)")
        except Exception as e:
            print(f"❌ [WebRTC] Failed to save audio: {e}")


def setup_webrtc_routes(app: FastAPI, audio_queue: Optional[multiprocessing.Queue] = None):
    """Set up WebRTC routes - direct port of aiortc server.py for FastAPI"""
    
    @app.post("/api/webrtc/offer")
    async def offer(params: dict):
        """Handle WebRTC offer - exact copy of aiortc server.py offer() function"""
        
        # Extract parameters - same as aiortc server.py
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        print(f"🤝 [WebRTC] Received offer")
        
        # Create peer connection - exact copy from aiortc server.py
        pc = RTCPeerConnection()
        pc_id = f"PeerConnection({id(pc)})"
        pcs.add(pc)
        
        print(f"🔧 [WebRTC] Created {pc_id}")
        
        # Use provided audio queue or create blackhole
        current_audio_queue = audio_queue or multiprocessing.Queue(maxsize=1000)
        
        # Set up recorder equivalent (our audio queue)
        recorder = MediaBlackhole()  # We don't record to file, just process
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """Monitor connection state - exact copy from aiortc server.py"""
            print(f"🔗 [WebRTC] {pc_id} connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                print(f"❌ [WebRTC] {pc_id} connection failed")
            elif pc.connectionState == "closed":
                print(f"🔌 [WebRTC] {pc_id} connection closed")
                pcs.discard(pc)
                await recorder.stop()

        @pc.on("track")
        def on_track(track):
            """Handle incoming track - adapted from aiortc server.py"""
            print(f"🎤 [WebRTC] {pc_id} received {track.kind} track")
            
            if track.kind == "audio":
                # Create audio processor track (replaces player.audio from server.py)
                audio_processor = AudioProcessorTrack(track, current_audio_queue, pc_id)
                pc.addTrack(audio_processor)
                
                print(f"✅ [WebRTC] {pc_id} audio processing started")
            
            @track.on("ended")
            async def on_ended():
                """Track ended handler - exact copy from aiortc server.py"""
                print(f"🔌 [WebRTC] {pc_id} track {track.kind} ended")
                await recorder.stop()

        # Set remote description - exact copy from aiortc server.py
        await pc.setRemoteDescription(offer)

        # Create answer - exact copy from aiortc server.py
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print(f"✅ [WebRTC] {pc_id} created answer")

        # Return response - exact copy from aiortc server.py format
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    @app.on_event("shutdown")
    async def on_shutdown():
        """Cleanup on shutdown - exact copy from aiortc server.py"""
        print("🧹 [WebRTC] Shutting down WebRTC connections...")
        
        # Close all peer connections
        coros = [pc.close() for pc in pcs]
        await asyncio.gather(*coros)
        pcs.clear()
        
        print("✅ [WebRTC] All connections closed")
    
    print("✅ [WebRTC] aiortc server routes registered")
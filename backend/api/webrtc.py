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
        # Convert to numpy array for STT processing
        audio_array = frame.to_ndarray()
        print(f"[DEBUG] Original shape: {frame.to_ndarray().shape}")
        # Assume always (1, N): flatten to 1D
        audio_array = audio_array.flatten()
        # Remove duplication: take every other sample
        audio_array = audio_array[::2]
        print(f"[DEBUG] After deduplication: {audio_array[:20]}")
        # Set sample rate and max_samples on first frame
        if self.sample_rate is None:
            # WebRTC audio from browsers is almost always 48kHz (Opus default)
            self.sample_rate = 48000
            self.max_samples = int(self.max_seconds * self.sample_rate)
        print(f"[DEBUG] After flatten: shape={audio_array.shape}, dtype={audio_array.dtype}, min={audio_array.min()}, max={audio_array.max()}, sample_rate={self.sample_rate}, sample={audio_array[:10]}")
        # Rolling buffer logic (only if max_samples is set)
        if self.max_samples is not None:
            self.rolling_buffer.append(audio_array)
            self.rolling_buffer_samples += len(audio_array)
            # Trim buffer to last 5 seconds
            while self.rolling_buffer_samples > self.max_samples:
                removed = self.rolling_buffer.popleft()
                self.rolling_buffer_samples -= len(removed)
        # Queue for STT processing
        try:
            self.audio_queue.put_nowait(audio_array)
            # Log first few frames
            if self.frame_count <= 3:
                print(f"🎤 [WebRTC] Frame {self.frame_count}: {audio_array.shape} samples, max: {np.max(np.abs(audio_array)):.4f}")
            elif self.frame_count % 50 == 0:
                print(f"🎤 [WebRTC] Processed {self.frame_count} frames")
        except Exception as e:
            print(f"❌ [WebRTC] Queue error: {e}")
        # Save last 5 seconds to file every 200 frames (for testing)
        if self.frame_count % 200 == 0:
            self.save_last_5_seconds_to_wav(f"audio_recording_{self.connection_id}_{self.frame_count}.wav")
        return frame
    
    def save_last_5_seconds_to_wav(self, filename):
        """Save the last 5 seconds of audio to a WAV file."""
        if self.sample_rate is None or self.rolling_buffer_samples == 0 or self.max_samples is None:
            print(f"[WebRTC] No audio to save.")
            return
        audio_data = np.concatenate(list(self.rolling_buffer))
        # Only keep the last max_samples
        if len(audio_data) > self.max_samples:
            audio_data = audio_data[-self.max_samples:]
        print(f"[DEBUG] Saving WAV: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min()}, max={audio_data.max()}, sample_rate={self.sample_rate}")
        try:
            sf.write(filename, audio_data, self.sample_rate, subtype='PCM_16')
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
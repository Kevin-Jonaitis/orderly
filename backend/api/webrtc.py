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
        
        print(f"🎧 [WebRTC] AudioProcessorTrack created for {connection_id}")
    
    async def recv(self):
        """Process audio frames from browser"""
        frame = await self.track.recv()
        
        self.frame_count += 1
        
        # Convert to numpy array for STT processing
        audio_array = frame.to_ndarray()
        
        # Handle multi-channel audio
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Ensure float32 format
        audio_array = audio_array.astype(np.float32)
        
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
        
        return frame


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
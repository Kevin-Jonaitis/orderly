"""
WebRTC server - direct port of aiortc server.py example for FastAPI
Adapted for audio-only streaming to STT processor
"""

import asyncio
import json
import multiprocessing
import time

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from fastapi import FastAPI, HTTPException
import collections
import soundfile as sf
import scipy.signal
from av import AudioFrame

# Import the AudioProcessorTrack from its new location
from processes.audio_processor_track import AudioProcessorTrack
# Import the AudioResponseTrack for outgoing audio
from processes.audio_response_track import AudioResponseTrack

# Global state - exact copy from aiortc server.py
pcs = set()



def setup_webrtc_routes(app: FastAPI, audio_queue: multiprocessing.Queue, audio_output_webrtc_queue: multiprocessing.Queue):
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
                audio_processor = AudioProcessorTrack(track, audio_queue, pc_id)
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
        
        # Add AudioResponseTrack for streaming audio to frontend
        audio_response_track = AudioResponseTrack(audio_output_webrtc_queue, pc_id)
        pc.addTransceiver(audio_response_track, direction="sendonly")
        print(f"🎵 [WebRTC] {pc_id} audio response streaming started")
        
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
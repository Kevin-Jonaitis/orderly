"""
WebRTC server - direct port of aiortc server.py example for FastAPI
Adapted for audio-only streaming to STT processor with very low latency jitter buffer settings
"""

import asyncio
import json
import multiprocessing
import time
import os

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from fastapi import FastAPI, HTTPException
import collections
from av import AudioFrame

# Import the AudioProcessorTrack from its new location
from processes.audio_processor_track import AudioProcessorTrack

# Global state - exact copy from aiortc server.py
pcs = set()

# Very Low Latency Jitter Buffer Configuration
# These settings map to the libwebrtc parameters:
# - jitter_buffer_max_packets -> capacity (must be power of 2)
# - jitter_buffer_min_delay_ms -> prefetch (lower = less buffering)
# - jitter_buffer_fast_playout -> reduced prefetch for lower latency
JITTER_BUFFER_CONFIG = {
    "audio": {
        "capacity": 2,        # Very low latency (was 16) - jitter_buffer_max_packets
        "prefetch": 0,        # No prefetch for immediate playout - jitter_buffer_min_delay_ms
        "is_video": False
    },
    "video": {
        "capacity": 32,       # Reduced from default 128 for lower latency
        "prefetch": 0,        # No prefetch for video (fast playout)
        "is_video": True
    }
}

def setup_custom_jitter_buffer():
    """Monkey patch RTCRtpReceiver to use custom jitter buffer settings"""
    from aiortc.rtcrtpreceiver import RTCRtpReceiver
    from aiortc.jitterbuffer import JitterBuffer
    
    # Store original __init__ method
    original_init = RTCRtpReceiver.__init__
    
    def custom_init(self, kind: str, transport):
        # Call original init
        original_init(self, kind, transport)
        
        # Override jitter buffer with custom settings
        if kind in JITTER_BUFFER_CONFIG:
            config = JITTER_BUFFER_CONFIG[kind]
            self._RTCRtpReceiver__jitter_buffer = JitterBuffer(
                capacity=config["capacity"],
                prefetch=config["prefetch"],
                is_video=config["is_video"]
            )
            print(f"🎯 [WebRTC] Applied very low latency jitter buffer for {kind}: capacity={config['capacity']}, prefetch={config['prefetch']}")
    
    # Replace the __init__ method
    RTCRtpReceiver.__init__ = custom_init
    print("✅ [WebRTC] Custom jitter buffer configuration applied globally")

def setup_webrtc_routes(app: FastAPI, audio_queue: multiprocessing.Queue, stt_warmup_flag):
    """Set up WebRTC routes with very low latency jitter buffer configuration"""
    
    # Apply custom jitter buffer configuration globally
    setup_custom_jitter_buffer()
    
    @app.post("/api/webrtc/offer")
    async def offer(params: dict):
        """Handle WebRTC offer with very low latency jitter buffer settings"""
        
        # Extract parameters
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        print(f"🤝 [WebRTC] Received offer")
        
        # Create peer connection (will automatically use custom jitter buffer)
        pc = RTCPeerConnection()
        pc_id = f"PeerConnection({id(pc)})"
        pcs.add(pc)
        
        print(f"🔧 [WebRTC] Created {pc_id} with very low latency jitter buffer settings")
        print(f"⚡ [WebRTC] Audio jitter buffer: capacity=4, prefetch=0 (ultra-low latency)")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """Monitor connection state and trigger STT warm-up when connected"""
            print(f"🔗 [WebRTC] {pc_id} connection state is {pc.connectionState}")
            
            if pc.connectionState == "connected":
                print("✅ [WebRTC] Connection established - triggering STT warm-up...")
                # Set the warm-up flag to signal the STT process
                stt_warmup_flag.value = 1
                print("🔥 [WebRTC] STT warm-up flag set")
                    
            elif pc.connectionState == "failed":
                print(f"❌ [WebRTC] {pc_id} connection failed")
            elif pc.connectionState == "closed":
                print(f"🔌 [WebRTC] {pc_id} connection closed")
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            """Handle incoming track with very low latency jitter buffer"""
            print(f"🎤 [WebRTC] {pc_id} received {track.kind} track")
            
            if track.kind == "audio":
                # Find the transceiver for this track
                for transceiver in pc.getTransceivers():
                    if transceiver.receiver.track == track:
                        mid = transceiver.mid
                        print(f"🎤 [WebRTC] {pc_id} found track in transceiver with MID: {mid}")
                        
                        # Only handle incoming audio (MID 0) - no outgoing audio via WebRTC
                        if mid == "0":
                            print(f"🎤 [WebRTC] {pc_id} setting up AudioProcessorTrack for MID 0")
                            audio_processor = AudioProcessorTrack(track, audio_queue, pc_id)
                            pc.addTrack(audio_processor)
                        else:
                            print(f"⚠️ [WebRTC] {pc_id} unexpected MID: {mid}")
                        break
            
            @track.on("ended")
            async def on_ended():
                """Track ended handler"""
                print(f"🔌 [WebRTC] {pc_id} track {track.kind} ended")

        # Set remote description
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print(f"✅ [WebRTC] {pc_id} created answer")
        
        # Debug: Check final transceiver state after answer
        transceivers = pc.getTransceivers()
        print(f"🎵 [WebRTC] {pc_id} FINAL transceivers after answer: {len(transceivers)}")
        for i, transceiver in enumerate(transceivers):
            print(f"🎵 [WebRTC] {pc_id} FINAL transceiver {i}: direction={transceiver.direction}, currentDirection={transceiver.currentDirection}, mid={transceiver.mid}")
            if transceiver.sender.track:
                print(f"🎵 [WebRTC] {pc_id} FINAL transceiver {i} sender track: {transceiver.sender.track.id}, readyState={transceiver.sender.track.readyState}")
            if transceiver.receiver.track:
                print(f"🎵 [WebRTC] {pc_id} FINAL transceiver {i} receiver track: {transceiver.receiver.track.id}, readyState={transceiver.receiver.track.readyState}")

        # Return response
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
    
    print("✅ [WebRTC] aiortc server routes registered with very low latency jitter buffer settings")
    print("⚡ [WebRTC] Audio jitter buffer: capacity=4, prefetch=0 (ultra-low latency mode)")
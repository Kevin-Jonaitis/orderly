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
        """Handle WebRTC offer - simplified based on aiortc server.py"""
        
        # Extract parameters
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        
        print(f"🤝 [WebRTC] Received offer")
        
        # Create peer connection
        pc = RTCPeerConnection()
        pc_id = f"PeerConnection({id(pc)})"
        pcs.add(pc)
        
        print(f"🔧 [WebRTC] Created {pc_id}")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """Monitor connection state"""
            print(f"🔗 [WebRTC] {pc_id} connection state is {pc.connectionState}")
            if pc.connectionState == "failed":
                print(f"❌ [WebRTC] {pc_id} connection failed")
            elif pc.connectionState == "closed":
                print(f"🔌 [WebRTC] {pc_id} connection closed")
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            """Handle incoming track - assign transceiver 0 to AudioProcessorTrack, transceiver 1 to AudioResponseTrack"""
            print(f"🎤 [WebRTC] {pc_id} received {track.kind} track")
            
            if track.kind == "audio":
                # Find the transceiver for this track
                for transceiver in pc.getTransceivers():
                    if transceiver.receiver.track == track:
                        mid = transceiver.mid
                        print(f"🎤 [WebRTC] {pc_id} found track in transceiver with MID: {mid}")
                        
                        # Assign MID 0 to AudioProcessorTrack, MID 1 to AudioResponseTrack
                        if mid == "0":
                            print(f"🎤 [WebRTC] {pc_id} setting up AudioProcessorTrack for MID 0")
                            audio_processor = AudioProcessorTrack(track, audio_queue, pc_id)
                            pc.addTrack(audio_processor)
                        elif mid == "1":
                            print(f"🎵 [WebRTC] {pc_id} setting up AudioResponseTrack for MID 1")
                            audio_response = AudioResponseTrack(audio_output_webrtc_queue, pc_id)
                            pc.addTrack(audio_response)
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
    
    print("✅ [WebRTC] aiortc server routes registered")
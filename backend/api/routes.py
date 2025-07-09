"""
FastAPI routes for the speech processing API
"""

import asyncio
import base64
import json
import multiprocessing
import queue
import time
import os
import subprocess
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

# WebRTC functions will be imported from main app setup

router = APIRouter()

# Global state for WebSocket connections
websocket_connections: Dict[str, WebSocket] = {}
connection_audio_queues: Dict[str, multiprocessing.Queue] = {}

# Global WebSocket audio queue (will be set by main app)
websocket_audio_queue: multiprocessing.Queue = None  # type: ignore

# Global order tracker (will be set by main app)
order_tracker = None  # type: ignore

# Global order update queue (will be set by main app)
order_update_queue = None  # type: ignore

# Global warmup queue (will be set by main app)
warmup_queue = None  # type: ignore

# Global set of active order WebSocket connections
order_websocket_connections = set()

# Menu upload and processing endpoints
@router.post("/upload-menu")
async def upload_menu(file: UploadFile = File(...)):
    """Upload a menu image file"""
    
    # Create menus directory if it doesn't exist
    menus_dir = Path("menus")
    menus_dir.mkdir(exist_ok=True)
    
    # Always save as menu.jpg
    filename = "menu.jpg"
    file_path = menus_dir / filename
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    print(f"📋 [Menu] Uploaded menu image: {filename}")
    return {"filename": filename, "status": "uploaded"}

@router.post("/process-menu")
async def process_menu(request: dict):
    """Process uploaded menu image with OCR"""
    
    filename = request.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")
    
    # Ensure menus directory exists
    menus_dir = Path("menus")
    image_path = menus_dir / filename
    
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Menu image not found")
    
    try:
        # Import and run the single processing function
        from OCRAndParseMenu import process_menu_image
        
        # Process the menu image
        success = process_menu_image(str(image_path))
        
        if not success:
            raise HTTPException(status_code=500, detail="Menu processing failed")
        
        # Send warmup signal to LLM process
        if warmup_queue:
            try:
                print(f"🔥 [Menu] About to send warmup signal to queue: {warmup_queue}")
                warmup_queue.put("warmup")
                print("🔥 [Menu] Sent warmup signal to LLM process")
            except Exception as e:
                print(f"⚠️ [Menu] Failed to send warmup signal: {e}")
        else:
            print("⚠️ [Menu] No warmup queue available")
        
        print(f"📋 [Menu] Successfully processed menu: {filename}")
        return {"status": "processed", "filename": filename}
        
    except Exception as e:
        print(f"❌ [Menu] OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@router.get("/current-menu-image")
async def get_current_menu_image():
    """Get the current menu image"""
    # Always return menu.jpg if it exists
    image_path = Path("menus") / "menu.jpg"
    if not image_path.exists():
        return {"image": None}
    
    return {"image": "menu.jpg"}

def set_websocket_audio_queue(queue: multiprocessing.Queue):
    """Set the global WebSocket audio queue"""
    global websocket_audio_queue
    websocket_audio_queue = queue

def set_order_tracker(tracker):
    """Set the global order tracker"""
    global order_tracker
    order_tracker = tracker

def set_order_update_queue(queue):
    """Set the global order update queue"""
    global order_update_queue
    order_update_queue = queue

def set_warmup_queue(queue):
    """Set the global warmup queue"""
    global warmup_queue
    warmup_queue = queue

# WebRTC endpoint is handled in webrtc.py

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for TTS audio streaming"""
    print(f"🎵 [WebSocket] New connection attempt...")
    await websocket.accept()
    connection_id = f"ws_{int(time.time() * 1000)}"
    
    print(f"🎵 [WebSocket] Client connected: {connection_id}")
    
    try:
        # Store connection
        websocket_connections[connection_id] = websocket
        
        # Use the global WebSocket audio queue (shared by all connections)
        if websocket_audio_queue is None:
            print("❌ [WebSocket] Global WebSocket audio queue not set!")
            await websocket.close()
            return
        
        print(f"🎵 [WebSocket] Using global audio queue for {connection_id}")
        
        # Start TTS audio streaming task
        streaming_task = asyncio.create_task(
            stream_tts_audio(websocket, websocket_audio_queue, connection_id)
        )
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "tts_start":
                    print(f"🎵 [WebSocket] TTS started for {connection_id}")
                elif message.get("type") == "tts_stop":
                    print(f"🎵 [WebSocket] TTS stopped for {connection_id}")
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"❌ [WebSocket] Error handling message: {e}")
                break
                
    except WebSocketDisconnect:
        print(f"🎵 [WebSocket] Client disconnected: {connection_id}")
    except Exception as e:
        print(f"❌ [WebSocket] Error in websocket endpoint: {e}")
    finally:
        # Cleanup
        if connection_id in websocket_connections:
            del websocket_connections[connection_id]
        
        # Cancel streaming task
        if 'streaming_task' in locals():
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass
        
        print(f"🎵 [WebSocket] Cleanup complete for {connection_id}")

async def stream_tts_audio(websocket: WebSocket, audio_queue: multiprocessing.Queue, connection_id: str):
    """Stream TTS audio chunks to WebSocket client with ultra-low latency optimizations"""
    print(f"🎵 [WebSocket] Starting TTS audio streaming for {connection_id}")
    chunk_count = 0
    empty_count = 0
    
    try:
        while True:
            # Check if WebSocket is still connected
            if websocket.client_state.value > 2:  # WebSocket is closed
                print(f"🎵 [WebSocket] WebSocket closed for {connection_id}")
                break
                
            try:
                # Get audio chunk from the connection's audio queue (non-blocking)
                audio_chunk = audio_queue.get_nowait()
                chunk_count += 1
                empty_count = 0  # Reset empty count
                
                # Convert to base64
                audio_bytes = audio_chunk.tobytes()
                base64_chunk = base64.b64encode(audio_bytes).decode('utf-8')
                
                # Send to client
                message = {
                    "type": "tts_chunk",
                    "content": base64_chunk
                }
                await websocket.send_text(json.dumps(message))
                
                # if chunk_count % 10 == 0:  # Log every 10 chunks
                #     print(f"🎵 [WebSocket] Sent {chunk_count} audio chunks to {connection_id}")
                
            except queue.Empty:
                # No audio data available, use ultra-low latency sleep (1ms like RealtimeVoiceChat)
                empty_count += 1
                if empty_count % 100 == 0:  # Log every 100 empty checks
                    print(f"🎵 [WebSocket] No audio data available for {connection_id} (empty count: {empty_count})")
                await asyncio.sleep(0.001)  # 1ms sleep for ultra-low latency
            except Exception as e:
                print(f"❌ [WebSocket] Error streaming audio: {e}")
                break
                
    except asyncio.CancelledError:
        print(f"🎵 [WebSocket] TTS streaming cancelled for {connection_id}")
    except Exception as e:
        print(f"❌ [WebSocket] Error in TTS streaming: {e}")
    finally:
        print(f"🎵 [WebSocket] TTS streaming ended for {connection_id}, sent {chunk_count} chunks")

def send_tts_audio_to_websocket(connection_id: str, audio_chunk):
    """Send TTS audio chunk to WebSocket client (called from TTS process)"""
    if connection_id in connection_audio_queues:
        try:
            connection_audio_queues[connection_id].put_nowait(audio_chunk)
        except Exception as e:
            print(f"❌ [WebSocket] Error sending audio to {connection_id}: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Order API endpoints
@router.get("/order")
async def get_order():
    """Get current order"""
    if order_tracker:
        order_data = order_tracker.format_order_for_frontend()
        return order_data
    else:
        return {"items": [], "total": 0}

@router.post("/order/clear")
async def clear_order():
    """Clear current order"""
    if order_tracker:
        order_tracker.clear_order()
        return {"status": "cleared"}
    else:
        return {"status": "error", "message": "Order tracker not available"}

@router.websocket("/ws/order")
async def order_websocket(websocket: WebSocket):
    """WebSocket endpoint for order updates"""
    await websocket.accept()
    connection_id = f"order_ws_{int(time.time() * 1000)}"
    print(f"📋 [Order] Client connected to order WebSocket: {connection_id}")
    order_websocket_connections.add(websocket)
    try:
        while True:
            # Wait for order data from the queue (blocking, but yields control)
            if order_update_queue is None:
                raise RuntimeError("order_update_queue must be set before using order WebSocket endpoints.")
            order_data = await asyncio.to_thread(order_update_queue.get)
            # Broadcast to all connected clients
            for ws in list(order_websocket_connections):
                try:
                    await ws.send_text(json.dumps(order_data))
                except Exception as e:
                    print(e)
    except WebSocketDisconnect:
        print(f"📋 [Order] Client disconnected from order WebSocket: {connection_id}")
    except Exception as e:
        print(f"❌ [Order] Error in order WebSocket: {e}")
    finally:
        order_websocket_connections.discard(websocket)


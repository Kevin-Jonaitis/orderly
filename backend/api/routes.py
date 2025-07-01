"""
API routes for the AI Order Taker backend.

This module provides:
- WebSocket endpoints for audio and order streaming
- HTTP endpoints for file uploads and order management
- Static file serving
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from processors.llm import OrderItem
from utils.latency import LatencyLogger
from utils.audio import convert_webm_to_wav

logger = logging.getLogger(__name__)

# Global state (in production, use proper state management)
current_order: List[OrderItem] = []
active_connections: List[WebSocket] = []
order_connections: List[WebSocket] = []
audio_accumulator: dict = {}  # Store accumulated audio per connection

# Initialize latency logger
latency_logger = LatencyLogger()

def setup_routes(app: FastAPI, stt_processor, llm_reasoner, tts_processor):
    """Set up all API routes for the FastAPI app"""
    
    @app.websocket("/ws/audio")
    async def websocket_audio(websocket: WebSocket):
        """WebSocket endpoint for audio streaming"""
        await websocket.accept()
        active_connections.append(websocket)
        connection_id = str(id(websocket))
        audio_accumulator[connection_id] = b""
        
        logger.info("Audio WebSocket connected")
        
        try:
            while True:
                # === TEST MODE: USE STATIC WAV FILE ===
                # Still receive bytes to maintain WebSocket protocol
                await websocket.receive_bytes()
                
                # Load test WAV file instead of processing real audio
                test_wav_path = "test/test_audio.wav"
                try:
                    with open(test_wav_path, "rb") as f:
                        wav_bytes = f.read()
                    
                    print(f"📁 Loading test file: {test_wav_path} ({len(wav_bytes)} bytes)")
                    
                    # Process test audio through STT pipeline
                    await process_audio_pipeline(wav_bytes, websocket, stt_processor, llm_reasoner, tts_processor)
                    
                    # Log test audio processing
                    latency_logger.log_event("AUDIO_RECEIVED", {
                        "test_file": test_wav_path,
                        "wav_size": len(wav_bytes)
                    })
                    
                except FileNotFoundError:
                    print(f"❌ Test file not found: {test_wav_path}")
                    print("💡 Create a test WAV file at test/test_audio.wav")
                    await asyncio.sleep(1)  # Prevent spam
                
        except WebSocketDisconnect:
            logger.info("Audio WebSocket disconnected")
            if websocket in active_connections:
                active_connections.remove(websocket)
            # Clean up accumulator
            if connection_id in audio_accumulator:
                del audio_accumulator[connection_id]
        except Exception as e:
            logger.error(f"Error in audio WebSocket: {e}")
            if websocket in active_connections:
                active_connections.remove(websocket)
            if connection_id in audio_accumulator:
                del audio_accumulator[connection_id]

    @app.websocket("/ws/order")
    async def websocket_order(websocket: WebSocket):
        """WebSocket endpoint for order updates"""
        await websocket.accept()
        order_connections.append(websocket)
        
        logger.info("Order WebSocket connected")
        
        try:
            # Send current order on connect
            order_data = {
                "items": [item.dict() for item in current_order],
                "total": sum(item.price * item.quantity for item in current_order)
            }
            await websocket.send_text(json.dumps(order_data))
            
            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                
        except WebSocketDisconnect:
            logger.info("Order WebSocket disconnected")
            if websocket in order_connections:
                order_connections.remove(websocket)

    async def process_audio_pipeline(audio_chunk: bytes, websocket: WebSocket, stt_processor, llm_reasoner, tts_processor):
        """Process audio through the full pipeline"""
        pipeline_start = time.time()
        
        try:
            # Step 1: STT
            transcribed_text = await stt_processor.transcribe(audio_chunk)
            
            # Step 2: LLM reasoning
            new_items = await llm_reasoner.process_order(transcribed_text)
            
            # Step 3: Update order
            global current_order
            current_order.extend(new_items)
            
            # Step 4: Generate response (using streaming method)
            from llama_cpp import RequestCancellation
            cancellation = RequestCancellation()
            response_text = ""
            async for token in llm_reasoner.generate_response_stream("TO CHANGE: MY TEXT", cancellation):
                response_text += token
            
            # Step 5: TTS
            audio_response = await tts_processor.synthesize(response_text)
            
            # Step 6: Send updates
            await broadcast_order_update()
            
            # Send transcription back
            try:
                await websocket.send_text(json.dumps({
                    "type": "transcription",
                    "text": transcribed_text
                }))
                
                # Send audio response (stub for now)
                await websocket.send_text(json.dumps({
                    "type": "audio_response",
                    "text": response_text
                }))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                raise  # Re-raise to trigger disconnection cleanup
            
            # Log total pipeline latency
            total_latency = (time.time() - pipeline_start) * 1000
            latency_logger.log_event("PIPELINE_TOTAL", {
                "transcription": transcribed_text,
                "response": response_text,
                "items_added": len(new_items)
            }, total_latency)
            
        except Exception as e:
            logger.error(f"Error in audio pipeline: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))

    async def broadcast_order_update():
        """Broadcast order updates to all connected clients"""
        order_data = {
            "items": [item.dict() for item in current_order],
            "total": sum(item.price * item.quantity for item in current_order)
        }
        
        # Send to all connected order WebSocket clients
        disconnected_connections = []
        for connection in order_connections:
            try:
                await connection.send_text(json.dumps(order_data))
            except Exception as e:
                logger.error(f"Error broadcasting to order connection: {e}")
                disconnected_connections.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected_connections:
            if connection in order_connections:
                order_connections.remove(connection)

    @app.post("/api/upload-menu")
    async def upload_menu(file: UploadFile = File(...)):
        """Upload menu file (text or image)"""
        try:
            # Save uploaded file
            file_path = Path("uploads") / file.filename
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Process based on file type
            if file.content_type.startswith("text/"):
                # Text file - save directly to menus
                menu_path = Path("menus") / f"{file.filename}.txt"
                with open(menu_path, "wb") as f:
                    f.write(content)
                
                logger.info(f"Saved text menu: {menu_path}")
                
            elif file.content_type.startswith("image/"):
                # Image file - stub OCR processing
                logger.info(f"Image uploaded: {file_path} (OCR processing stubbed)")
                
                # Stub: In real implementation, use OCR to extract text
                extracted_text = "Menu extracted from image (stub)"
                menu_path = Path("menus") / f"{file.filename}.txt"
                with open(menu_path, "w") as f:
                    f.write(extracted_text)
            
            # Reload menu context
            llm_reasoner.menu_context = llm_reasoner.load_menu_context()
            
            return {"message": "Menu uploaded successfully", "filename": file.filename}
            
        except Exception as e:
            logger.error(f"Error uploading menu: {e}")
            return {"error": str(e)}

    @app.get("/api/order")
    async def get_current_order():
        """Get current order"""
        return {
            "items": [item.dict() for item in current_order],
            "total": sum(item.price * item.quantity for item in current_order)
        }

    @app.post("/api/order/clear")
    async def clear_order():
        """Clear current order"""
        global current_order
        current_order = []
        await broadcast_order_update()
        return {"message": "Order cleared"}

    @app.get("/")
    async def root():
        """Serve the React app in production, API info in development"""
        # Check if we have static files (production mode)
        static_dir = Path("static")
        if static_dir.exists() and (static_dir / "index.html").exists():
            return FileResponse("static/index.html")
        else:
            # Development mode - just return API info
            return {"message": "AI Order Taker Backend - Development Mode"}

    # Catch-all route for React Router (must be last)
    @app.get("/{path:path}")
    async def serve_react_app(path: str):
        """Serve React app for any non-API route"""
        static_dir = Path("static")
        if static_dir.exists() and (static_dir / "index.html").exists():
            # Try to serve the specific file first
            file_path = static_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # Otherwise serve index.html (React Router will handle routing)
            return FileResponse("static/index.html")
        else:
            # Development mode - return 404
            raise HTTPException(status_code=404, detail="Not found")
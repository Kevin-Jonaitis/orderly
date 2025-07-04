"""
API routes for the AI Order Taker backend.

This module provides:
- WebSocket endpoints for order streaming
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

logger = logging.getLogger(__name__)

# Global state (in production, use proper state management)
current_order: List[OrderItem] = []
order_connections: List[WebSocket] = []

# Initialize latency logger
latency_logger = LatencyLogger()

def setup_routes(app: FastAPI, stt_processor, llm_reasoner, tts_processor):
    """Set up all API routes for the FastAPI app"""
    

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
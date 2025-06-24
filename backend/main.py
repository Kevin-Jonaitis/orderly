"""
AI Order Taker Backend
FastAPI + WebSocket server for real-time audio streaming and order processing
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data models
class OrderItem(BaseModel):
    id: str
    name: str
    price: float
    quantity: int = 1

class Order(BaseModel):
    items: List[OrderItem]
    total: float
    timestamp: datetime

# Global state (in production, use proper state management)
current_order: List[OrderItem] = []
active_connections: List[WebSocket] = []

# Create directories
Path("logs").mkdir(exist_ok=True)
Path("menus").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

app = FastAPI(title="AI Order Taker Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for production)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

class LatencyLogger:
    """Simple latency logging utility"""
    
    def __init__(self):
        self.log_file = Path("logs") / f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_event(self, event_type: str, data: Dict[str, Any], latency_ms: float = None):
        """Log an event with optional latency"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "latency_ms": latency_ms,
            "data": data
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"{event_type}: {latency_ms}ms" if latency_ms else f"{event_type}")

latency_logger = LatencyLogger()

# Stub components for the full pipeline
class STTProcessor:
    """Stub for Parakeet STT"""
    
    async def transcribe(self, audio_chunk: bytes) -> str:
        """Stub transcription - replace with Parakeet"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock transcription
        text = "I want a cheeseburger and fries"
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("STT_TRANSCRIBE", {"text": text}, latency_ms)
        
        return text

class LLMReasoner:
    """Stub for Phi-3 Mini reasoning LLM"""
    
    def __init__(self):
        self.menu_context = self.load_menu_context()
    
    def load_menu_context(self) -> str:
        """Load menu context from uploaded files"""
        menu_files = list(Path("menus").glob("*.txt"))
        if not menu_files:
            return "Default menu: Cheeseburger ($8.99), Fries ($3.99), Drink ($2.99)"
        
        context = ""
        for file in menu_files:
            context += file.read_text() + "\n"
        return context
    
    async def process_order(self, text: str) -> List[OrderItem]:
        """Process user text into order items"""
        start_time = time.time()
        
        # Simulate LLM processing
        await asyncio.sleep(0.2)
        
        # Mock order processing
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("LLM_REASONING", {
            "input_text": text,
            "output_items": [item.dict() for item in items]
        }, latency_ms)
        
        return items
    
    async def generate_response(self, order_items: List[OrderItem]) -> str:
        """Generate response text for TTS"""
        start_time = time.time()
        
        await asyncio.sleep(0.1)
        
        response = f"I've added {len(order_items)} items to your order. Anything else?"
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("LLM_RESPONSE", {
            "response_text": response
        }, latency_ms)
        
        return response

class TTSProcessor:
    """Stub for Chatterbox TTS"""
    
    async def synthesize(self, text: str) -> bytes:
        """Stub TTS synthesis"""
        start_time = time.time()
        
        await asyncio.sleep(0.15)
        
        # Mock audio data
        audio_data = b"mock_audio_data"
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("TTS_SYNTHESIS", {
            "text": text,
            "audio_length": len(audio_data)
        }, latency_ms)
        
        return audio_data

# Initialize processors
stt_processor = STTProcessor()
llm_reasoner = LLMReasoner()
tts_processor = TTSProcessor()

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket endpoint for audio streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    logger.info("Audio WebSocket connected")
    
    try:
        while True:
            # Receive audio chunk
            audio_chunk = await websocket.receive_bytes()
            
            # Log audio received
            latency_logger.log_event("AUDIO_RECEIVED", {
                "chunk_size": len(audio_chunk)
            })
            
            # Process audio through pipeline
            await process_audio_pipeline(audio_chunk, websocket)
            
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
        active_connections.remove(websocket)

@app.websocket("/ws/order")
async def websocket_order(websocket: WebSocket):
    """WebSocket endpoint for order updates"""
    await websocket.accept()
    
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

async def process_audio_pipeline(audio_chunk: bytes, websocket: WebSocket):
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
        
        # Step 4: Generate response
        response_text = await llm_reasoner.generate_response(new_items)
        
        # Step 5: TTS
        audio_response = await tts_processor.synthesize(response_text)
        
        # Step 6: Send updates
        await broadcast_order_update()
        
        # Send transcription back
        await websocket.send_text(json.dumps({
            "type": "transcription",
            "text": transcribed_text
        }))
        
        # Send audio response (stub for now)
        await websocket.send_text(json.dumps({
            "type": "audio_response",
            "text": response_text
        }))
        
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
    
    # In a real app, you'd have separate order WebSocket connections
    # For now, we'll let the frontend poll or use a separate connection

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
    return {"message": "Order cleared"}

@app.get("/")
async def root():
    """Serve the React app in production, API info in development"""
    # Check if we have static files (production mode)
    static_dir = Path("static")
    if static_dir.exists() and (static_dir / "index.html").exists():
        from fastapi.responses import FileResponse
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
        from fastapi.responses import FileResponse
        # Try to serve the specific file first
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        # Otherwise serve index.html (React Router will handle routing)
        return FileResponse("static/index.html")
    else:
        # Development mode - return 404
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
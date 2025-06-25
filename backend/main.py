"""
AI Order Taker Backend
FastAPI + WebSocket server for real-time audio streaming and order processing
"""

import asyncio
import json
import logging
import subprocess
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
order_connections: List[WebSocket] = []
audio_accumulator: Dict[str, bytes] = {}  # Store accumulated audio per connection

# Create directories
Path("logs").mkdir(exist_ok=True)
Path("menus").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("audio_debug").mkdir(exist_ok=True)

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

def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """Convert WebM audio bytes to WAV format for STT processing"""
    try:
        # Add ffmpeg flags to handle fragmented/incomplete WebM
        result = subprocess.run([
            'ffmpeg', 
            '-hide_banner', '-loglevel', 'error',  # Reduce noise
            '-i', 'pipe:0',           # Read from stdin
            '-f', 'wav',              # Output WAV format
            '-ac', '1',               # Mono audio
            '-ar', '16000',           # 16kHz sample rate
            '-acodec', 'pcm_s16le',   # 16-bit PCM
            '-fflags', '+genpts',     # Generate timestamps for incomplete streams
            '-avoid_negative_ts', 'make_zero',  # Handle timing issues
            'pipe:1'                  # Write to stdout
        ], 
        input=webm_bytes, 
        capture_output=True,
        check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Log more details about the failure
        stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr"
        logger.error(f"FFmpeg conversion failed (exit {e.returncode}): {stderr_output}")
        return b""

# STT components
import tempfile
from faster_whisper import WhisperModel

class STTProcessor:
    """Real-time STT using Faster-Whisper"""
    
    def __init__(self):
        # GPU-only implementation with optimized settings
        import os
        
        # Suppress various warning outputs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info("Loading Faster-Whisper model (base.en, GPU optimized)")
        self.model = WhisperModel(
            "base.en", 
            device="cuda", 
            compute_type="float16",
            device_index=0,
            cpu_threads=0  # Force GPU-only processing
        )
        logger.info("✅ Faster-Whisper GPU model loaded successfully")
        self.device = "GPU"
        
        # Warm up the model with a dummy inference
        logger.info("🔥 Warming up GPU model...")
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model with actual audio file to avoid cold start"""
        warmup_file = "test/warm_up.wav"
        
        try:
            start_time = time.time()
            segments, _ = self.model.transcribe(
                warmup_file,
                beam_size=1,
                best_of=1,
                temperature=0,
                condition_on_previous_text=False,
                word_timestamps=False,
                language="en",
                task="transcribe",
                vad_filter=False,
                vad_parameters=None
            )
            # Force evaluation of all segments
            list(segments)
            warmup_ms = (time.time() - start_time) * 1000
            
            logger.info(f"🚀 GPU warmup completed with {warmup_file} in {warmup_ms:.0f}ms")
            
        except FileNotFoundError:
            logger.warning(f"Warmup file not found: {warmup_file}, using fallback warmup")
            # Fallback to simple dummy warmup
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                import numpy as np
                import wave
                with wave.open(tmp_file.name, 'w') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(np.zeros(16000, dtype=np.int16).tobytes())
                
                start_time = time.time()
                segments, _ = self.model.transcribe(tmp_file.name, beam_size=1)
                list(segments)
                warmup_ms = (time.time() - start_time) * 1000
                
                import os
                os.unlink(tmp_file.name)
                
            logger.info(f"🚀 GPU fallback warmup completed in {warmup_ms:.0f}ms")
    
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav_bytes)
            tmp_file.flush()
            
            # Time the complete inference process
            inference_start = time.time()
            segments, info = self.model.transcribe(
                tmp_file.name,
                beam_size=1,           # Fastest decoding
                best_of=1,             # No multiple candidates
                temperature=0,         # Deterministic output
                condition_on_previous_text=False,  # No context carryover
                word_timestamps=False, # Skip word-level timestamps
                language="en",
                task="transcribe",
                vad_filter=False,      # Skip voice activity detection
                vad_parameters=None
            )
            
            # Process segments (where the real work happens)
            text = " ".join([segment.text.strip() for segment in segments])
            inference_ms = (time.time() - inference_start) * 1000
            
        # Cleanup
        import os
        os.unlink(tmp_file.name)
        
        # Calculate performance metrics
        duration_seconds = len(wav_bytes) / (16000 * 2)  # Approximate duration
        realtime_factor = duration_seconds * 1000 / inference_ms
        
        print(f"🎤 STT: {inference_ms:.0f}ms ({realtime_factor:.1f}x realtime) → '{text}'")
        
        latency_logger.log_event("STT_TRANSCRIBE", {
            "text": text, 
            "inference_ms": inference_ms,
            "realtime_factor": realtime_factor
        }, inference_ms)
        
        return text.strip()

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
    connection_id = str(id(websocket))
    audio_accumulator[connection_id] = b""
    
    logger.info("Audio WebSocket connected")
    
    try:
        while True:
            # === ORIGINAL AUDIO PROCESSING (COMMENTED OUT FOR TESTING) ===
            # # Receive audio chunk
            # webm_chunk = await websocket.receive_bytes()
            # 
            # # Accumulate chunks
            # audio_accumulator[connection_id] += webm_chunk
            # accumulated_audio = audio_accumulator[connection_id]
            # 
            # print(f"Accumulating... Total WebM bytes: {len(accumulated_audio)}")
            # 
            # # Try to convert accumulated audio every 3 chunks (to reduce processing load)
            # chunk_count = len(accumulated_audio) // 16422  # Approximate chunks received
            # 
            # if chunk_count > 0 and chunk_count % 3 == 0:  # Every 3 chunks
            #     wav_chunk = convert_webm_to_wav(accumulated_audio)
            #     
            #     if wav_chunk:
            #         # Save WAV chunk for debugging
            #         timestamp = int(time.time() * 1000)
            #         filename = f"audio_accumulated_{timestamp}.wav"
            #         with open(f"audio_debug/{filename}", "wb") as f:
            #             f.write(wav_chunk)
            #         
            #         print(f"CONVERSION SUCCESS: WebM {len(accumulated_audio)} bytes → WAV {len(wav_chunk)} bytes")
            #         print(f"WAV header: {wav_chunk[:12]}")
            #         
            #         # Process audio through pipeline
            #         await process_audio_pipeline(wav_chunk, websocket)
            #     else:
            #         print(f"Conversion failed for {len(accumulated_audio)} bytes")
            # 
            # # Log audio received
            # latency_logger.log_event("AUDIO_RECEIVED", {
            #     "chunk_size": len(webm_chunk),
            #     "accumulated_size": len(accumulated_audio),
            #     "estimated_chunks": chunk_count
            # })
            
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
                await process_audio_pipeline(wav_bytes, websocket)
                
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
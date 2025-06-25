"""
AI Order Taker Backend
FastAPI + WebSocket server for real-time audio streaming and order processing
"""

import logging
import signal
import sys
import multiprocessing
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import processors
from processors.stt import create_stt_processor
from processors.llm import LLMReasoner
from processors.tts import TTSProcessor

# Import API routes
from api.routes import setup_routes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# ================== STT MODEL SELECTION ==================
# Change this variable to switch between STT models
STT_MODEL = "parakeet"  # Options: "whisper", "parakeet"
# ==========================================================

# Initialize processors
logger.info("🚀 Initializing AI Order Taker processors...")
stt_processor = create_stt_processor(STT_MODEL)
llm_reasoner = LLMReasoner()
tts_processor = TTSProcessor()

# Set up all API routes
setup_routes(app, stt_processor, llm_reasoner, tts_processor)

logger.info("✅ AI Order Taker Backend initialized successfully")

def signal_handler(signum, frame):
    """Kill all child processes on Ctrl+C"""
    logger.info(f"🛑 Received signal {signum}, killing all processes...")
    
    # Kill all child processes
    for child in multiprocessing.active_children():
        logger.info(f"Killing child process: {child.pid}")
        child.terminate()
        child.join(timeout=1)
        if child.is_alive():
            child.kill()
    
    # Clear GPU memory cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cache cleared")
    except ImportError:
        pass
    
    logger.info("👋 Backend shutdown complete")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable auto-reload to prevent multiple model loads
        log_level="info"
    )
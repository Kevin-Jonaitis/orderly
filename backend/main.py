"""
AI Order Taker Backend
FastAPI + WebSocket server for real-time audio streaming and order processing
"""

import logging
import signal
import sys
import time
import multiprocessing
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import processors
from processors.stt import create_stt_processor
from processors.llm import LLMReasoner
from processors.orpheus_tts import OrpheusTTS

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

# Placeholder variables for processors (will be initialized in main)
stt_processor = None
llm_reasoner = None
tts_processor = None

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
    # ================== STT MODEL SELECTION ==================
    # Change this variable to switch between STT models
    STT_MODEL = "realtime"  # Only realtime supported
    # ==========================================================
    
    # Initialize processors with sequential loading and delays
    logger.info("🚀 Initializing AI Order Taker processors...")
    
    # Step 1: Load STT processor (CPU-based, no GPU conflict)
    logger.info("📝 Loading STT processor...")
    stt_processor = create_stt_processor(STT_MODEL)
    logger.info("✅ STT processor loaded successfully")
    
    # Step 2: Delay + CUDA cleanup before LLM
    logger.info("⏳ Waiting 20 seconds before loading LLM...")
    # time.sleep(20)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"🧹 CUDA cache cleared. GPU memory: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
    
    # Step 3: Load LLM processor
    logger.info("🧠 Loading LLM processor...")
    llm_reasoner = LLMReasoner()
    logger.info("✅ LLM processor loaded successfully")
    
    # Step 4: Delay + CUDA cleanup before TTS
    logger.info("⏳ Waiting 20 seconds before loading TTS...")
    # time.sleep(20)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"🧹 CUDA cache cleared. GPU memory: {torch.cuda.memory_allocated() / (1024*1024):.1f}MB")
    
    # Step 5: Load TTS processor
    logger.info("🎵 Loading TTS processor...")
    tts_processor = OrpheusTTS()
    logger.info("✅ TTS processor loaded successfully")
    
    # Set up all API routes
    setup_routes(app, stt_processor, llm_reasoner, tts_processor)
    
    logger.info("✅ AI Order Taker Backend initialized successfully")

    # Test that the STT processor is still working and loaded; I'm not sure how llama handles multiple models
    
    # Load test audio file
    test_wav_path = Path("backend/test/test_audio.wav")
    if test_wav_path.exists():
        with open(test_wav_path, "rb") as f:
            wav_bytes = f.read()
        
        # Run async transcribe in sync context using asyncio.run()
        import asyncio
        result = asyncio.run(stt_processor.transcribe(wav_bytes))
        print("THE RESULT:")
        print(result)
    else:
        logger.warning(f"Test audio file not found: {test_wav_path}")
    
    # Use uvicorn.Server to avoid re-importing the module
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()
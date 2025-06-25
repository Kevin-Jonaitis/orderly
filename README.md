# AI Order Taker

A real-time AI-powered order taking system with streaming audio and intelligent menu processing.

## Architecture

- **Frontend**: Vite + React + TypeScript
- **Backend**: FastAPI + WebSocket streaming
- **Audio Pipeline**: Browser → WebSocket → Parakeet STT → Phi-3 LLM → Chatterbox TTS → SNAC → Browser

## Features

### Current Implementation
- ✅ Real-time audio streaming from browser to backend via WebSocket
- ✅ Live order display with automatic updates
- ✅ Menu upload (text/image) with backend processing
- ✅ Comprehensive latency logging system
- ✅ Stubbed pipeline components ready for LLM integration

### Planned Integration
- 🔄 Parakeet STT for speech recognition
- 🔄 Phi-3 Mini for order reasoning
- 🔄 Chatterbox + SNAC for text-to-speech
- 🔄 GPU optimization for 12GB RTX 4070

## Quick Start

### System Requirements
```bash
# Install system dependencies (Ubuntu/Debian/WSL)
sudo apt update && sudo apt install -y ffmpeg

# For other systems:
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### First Time Setup
```bash
# Setup virtual environment and install all dependencies
python3 setup_env.py
```

### Development Mode (Hot Reload)
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Option 1: Backend only (API at port 8000)
cd backend && python main.py

# Option 2: Full stack with hot reload
# Terminal 1 - Backend
cd backend && python main.py
# Terminal 2 - Frontend (new terminal, activate venv again)
source venv/bin/activate && cd frontend && npm run dev
```
Visit http://localhost:5173 (full stack) or http://localhost:8000 (backend only)

### Production Mode (Single Server)
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# OR  
venv\Scripts\activate     # Windows

# Build and run
python3 build.py
python3 run_production.py
```
Visit http://localhost:8000

### Lazy Mode (Auto-managed venv)
```bash
# These scripts auto-detect and use venv if available
python3 run_dev.py           # Development
python3 start_production.py  # Production
```

## Performance Targets

- **Latency Goal**: <700ms end-to-end
- **Hardware**: 12GB RTX 4070
- **Streaming**: Real-time audio chunks (100ms intervals)

## Logging

All latency measurements are automatically logged to `backend/logs/latency_*.log` with detailed pipeline timing.

## API Endpoints

- `ws://localhost:8000/ws/audio` - Audio streaming WebSocket
- `POST /api/upload-menu` - Upload menu files
- `GET /api/order` - Get current order
- `POST /api/order/clear` - Clear current order

## Development

The system is designed for easy LLM swapping. Each component (STT, LLM, TTS) is isolated and can be replaced independently.
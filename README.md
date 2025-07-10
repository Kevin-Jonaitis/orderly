# AI Order Taker

Demo of menu upload + ordering:

https://github.com/user-attachments/assets/b3643b8d-9fab-4ae9-9768-ca443a2dce16

Another demo of ordering with a seperate menu:

https://github.com/user-attachments/assets/2f1602ee-8ee4-421c-ac9d-9a9dd7112df9



A real-time AI-powered voice order taking system with streaming audio, intelligent menu processing, and live order management.

## 🎯 Overview

This system allows customers to place orders through natural voice conversation with an AI that understands menu items, processes modifications, and maintains order state in real-time.

## 🏗️ Architecture

- **Frontend**: React + TypeScript + Bootstrap (Vite dev server)
- **Backend**: FastAPI + WebSocket + WebRTC
- **Audio Pipeline**: Browser WebRTC → STT (Whisper) → Phi-3 LLM → TTS (Orpheus) → Browser
- **Order Management**: Real-time order tracking with WebSocket updates
- **Each pipeline stage (STT, LLM, TTS, Audio) runs in its own process, communicating via multiprocessing queues for robust, real-time streaming.**

## ✨ Features

### 🎤 Voice Interface
- **Real-time audio streaming** via WebRTC from browser to backend
- **Speech-to-Text** using Whisper with GPU acceleration
- **Natural language processing** with Phi-3 Medium LLM
- **Text-to-Speech** using Orpheus model for AI responses
- **Ultra-low latency** audio pipeline optimized for real-time conversation

### 🎯 Order Management
- **Live order display** with automatic WebSocket updates
- **Order modifications** (add, remove, change quantities)
- **Price calculation** with real-time totals
- **Order persistence** across conversation turns

### 🍽️ Menu System
- **Menu upload** via web interface (image/text)
- **OCR processing** for menu image analysis
- **AI-powered menu parsing** with structured item extraction
- **Dynamic menu updates** without system restart

### 🎛️ User Interface
- **Simple voice activation** - "Click to Start Order" button
- **Real-time transcription** display
- **Live order summary** with itemized list and totals
- **Menu image display** for customer reference

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)
- FFmpeg installed
- Node.js 18+ (for frontend)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd orderly

# Create virtual environment
python -m venv venv312
source venv312/bin/activate  # On Windows: venv312\Scripts\activate
```

2. **Install llama-cpp-python with CUDA support:**
```bash
# Build llama-cpp-python with CUDA support for GPU acceleration
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install remaining Python dependencies
pip install -r backend/requirements.txt
```

3. **Download AI models:**
```bash
# Create models directory
mkdir -p models

# Download Phi-3 Medium LLM (required)
# Place Phi-3-medium-4k-instruct.gguf in models/ directory

# Download Orpheus TTS model (required)
# Place orpheus model files in models/ directory
```

4. **Setup frontend:**
```bash
cd frontend
npm install
```

5. **Configure API keys (optional):**
```bash
# For OpenAI menu analysis (optional)
echo "your-openai-api-key" > backend/open-ai-api-key.txt
```

### Running the System

1. **Start the backend:**
```bash
cd backend
python multiprocess_stt_llm_tts.py
```

2. **Start the frontend (in new terminal):**
```bash
cd frontend
npm run dev
```

3. **Access the application:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8002

## 📁 Project Structure

```
orderly/
├── backend/
│   ├── api/                  # FastAPI routes and WebRTC integration
│   ├── processes/            # Multiprocessing pipeline (STT, LLM, TTS, Audio)
│   ├── processors/           # Model wrappers (Whisper, Orpheus, LLM, etc.)
│   ├── utils/                # Utility modules (order tracking, etc.)
│   ├── menus/                # Uploaded and processed menu images/text
│   ├── prompts/              # Prompt templates for LLM
│   ├── uploads/              # Temporary uploads (ignored)
│   ├── STT_debug_audio/      # Debug audio files (ignored)
│   ├── logs/                 # Backend logs (ignored)
│   ├── multiprocess_stt_llm_tts.py  # Main backend entrypoint
│   ├── OCRAndParseMenu.py    # OCR and menu parsing logic
│   ├── decoder.py            # Audio decoding utilities
│   └── requirements.txt      # Backend Python dependencies
├── frontend/
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # React components (MenuUpload, OrderDisplay, etc.)
│   │   ├── hooks/            # Custom React hooks
│   │   ├── types/            # TypeScript types
│   │   └── App.tsx           # Main React app
│   ├── package.json          # Frontend dependencies
│   ├── vite.config.ts        # Vite config
│   └── ...                   # Other config files
├── models/                   # AI models (Phi-3, Orpheus, etc.)
├── .gitignore
├── README.md
└── ... (other dev, venv, and external directories)
```

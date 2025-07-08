# AI Order Taker

A real-time AI-powered order taking system with streaming audio and intelligent menu processing.

## Architecture

- **Frontend**: Vite + React + TypeScript
- **Backend**: FastAPI + WebSocket streaming
- **Audio Pipeline**: Browser → WebSocket → STT (Whisper/Parakeet) → Phi-3 LLM → Chatterbox TTS → SNAC → Browser

## Features

### Current Implementation
- ✅ Real-time audio streaming from browser to backend via WebSocket
- ✅ Live order display with automatic updates
- ✅ Menu upload (text/image) with backend processing
- ✅ Comprehensive latency logging system
- ✅ Stubbed pipeline components ready for LLM integration
- ✅ OCR menu reading with EasyOCR

### Current STT Implementation
- ✅ **Whisper STT**: Faster-Whisper with GPU optimization (tiny.en, int8)
- ✅ **Parakeet STT**: NeMo FastConformer (optional)
- ✅ **Modular design**: Easy switching between STT models
- ✅ **Performance**: Sub-200ms inference on RTX 3070

### Current LLM Implementation
- ✅ **Phi-3 Mini**: Order reasoning with llama.cpp (Q4 quantized)
- ✅ **GPU Acceleration**: CUDA-optimized inference (~1s response time)
- ✅ **KV Cache**: Enabled for faster inference
- ✅ **Configurable prompts**: Easy system prompt modification

### OCR Implementation
- ✅ **EasyOCR**: Text extraction from menu images
- ✅ **Multi-language support**: English language detection
- ✅ **Confidence scoring**: Text confidence levels and bounding boxes
- ✅ **Batch processing**: Process multiple text blocks from single image

### Planned Integration  
- 🔄 Chatterbox + SNAC for text-to-speech

## Quick Start

### System Requirements
```bash
# Install system dependencies (Ubuntu/Debian/WSL)
sudo apt update && sudo apt install -y ffmpeg


# Instal torch w/ cuda(change the cu128 depending on your cuda version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install cuda-python==12.8

# For other systems:
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

### STT Model Selection
**Switch between STT models** by editing `backend/main.py` line 122:
```python
STT_MODEL = "whisper"    # Use Whisper (default, no extra deps)
STT_MODEL = "parakeet"   # Use Parakeet (requires NeMo installation)
```

**For Parakeet STT**, install additional dependencies:
```bash
pip install nemo_toolkit[asr] omegaconf hydra-core
```

### First Time Setup
```bash
# Setup virtual environment and install all dependencies
python3 setup_env.py

# Download Phi-3 Mini model (required for LLM)
mkdir -p models
curl -L -o models/Phi-3-mini-4k-instruct-q4.gguf https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf
```

### OCR Menu Reading
The system includes an OCR script to read text from menu images using EasyOCR.

**Setup:**
```bash
# Install OCR dependencies (included in requirements.txt)
pip install easyocr>=1.7.0

# Create menus directory
mkdir -p backend/menus
```

**Usage:**
1. Place menu images in `backend/menus/` directory
2. Edit `IMAGE_FILENAME` variable in `backend/ocr_reader.py` to specify which file to read
3. Run the OCR script:
```bash
cd backend
python ocr_reader.py
```

**Example Output:**
```
🔍 OCR Menu Reader
==================================================
🔍 Initializing EasyOCR...
✅ EasyOCR initialized successfully
📖 Reading text from: backend/menus/menu.png

📋 Found 3 text blocks:
==================================================
1. Text: 'Bean Burrito'
   Confidence: 0.95
   Bounding Box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
------------------------------
2. Text: '$8.99'
   Confidence: 0.92
   Bounding Box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
------------------------------
3. Text: 'Chicken Quesadilla'
   Confidence: 0.88
   Bounding Box: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
------------------------------

📄 Complete Text:
==================================================
Bean Burrito
$8.99
Chicken Quesadilla

✅ Successfully read 3 text blocks from menu.png
```

**Supported Image Formats:**
- PNG, JPG, JPEG, BMP, TIFF, and other common formats
- Works best with clear, high-contrast text
- Supports multiple text blocks per image

### GPU Setup for LLM Acceleration

## IMPORTANT: Python Environment for LLM Builds

- **Always use the `venv312` virtual environment for all llama-cpp-python builds and LLM-related dependencies.**
- Activate with:
  - `source venv312/Scripts/activate` (Windows, Git Bash)
  - Or use the full path: `../venv312/Scripts/python ...` for all pip/python commands

### ⚠️ GPU Support Warning (Python 3.12/Windows)
- As of June 2024, llama-cpp-python does **not** have official CUDA/cuBLAS GPU support for Python 3.12 on Windows.
- You may encounter linker errors (LNK1104) or failed builds when attempting to build with CUDA.
- For GPU support, use Python 3.10 or 3.11 in a separate venv (see below for instructions).
- If you must use Python 3.12, you are limited to CPU-only for llama-cpp-python until upstream support improves.

#### Troubleshooting: LNK1104 or Build Failures
- If you see errors like `LNK1104: cannot open file ... .exp` or Cython build failures:
  - Restart your shell and try again.
  - Ensure you are using Python 3.10/3.11 for GPU builds.
  - If using Python 3.12, expect CPU-only support for now.

#### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 11.8 or 12.x recommended)
- Python 3.10 or 3.11 (Python 3.13 may have compatibility issues)

#### Step 1: Verify CUDA Installation
```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi
```

#### Step 2: Uninstall CPU-only Version
```bash
pip uninstall llama-cpp-python -y
```

#### Step 3: Install Dependencies (if needed)
```bash
# For Python 3.10/3.11 (recommended)
pip install numpy==1.26.4 cython==3.0.10

# For Python 3.13 (may have issues)
pip install numpy==1.26.4 cython==3.0.10
```

#### Step 4: Build llama-cpp-python with CUDA Support
```bash
# Option A: Automatic build (recommended)
pip install llama-cpp-python --force-reinstall --upgrade --no-binary :all:

# Option B: Manual build with specific CUDA version
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Option C: Clone and build manually
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
pip install .
```

#### Step 5: Verify GPU Support
```python
import llama_cpp
print(llama_cpp.llama_cpp.llama_backend_version())
# Should show CUDA/cuBLAS information when loading a model
```

#### Step 6: Load Model with GPU Layers
In your code, make sure to specify `n_gpu_layers`:
```python
from llama_cpp import Llama
llm = Llama(
    model_path="path/to/model.gguf",
    n_gpu_layers=40,  # or as many as your GPU can handle
)
```

#### Troubleshooting

**If you see "all layers assigned to CPU" in logs:**
1. Verify llama-cpp-python was built with CUDA support
2. Check that `n_gpu_layers` is specified when loading the model
3. Ensure CUDA toolkit is properly installed and in PATH
4. Try restarting your Python environment

**If build fails on Windows with Python 3.13:**
1. Use Python 3.10 or 3.11 instead
2. Pre-install numpy and cython before building
3. Restart your shell/computer if you see linker errors

**Common Error Messages:**
- `LNK1104: cannot open file ... .exp` → Restart shell/computer, try Python 3.10/3.11
- `CUDA not available` → Check CUDA installation and PATH
- `No CUDA GPUs are available` → Check GPU drivers and nvidia-smi

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
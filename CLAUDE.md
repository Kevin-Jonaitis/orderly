# Claude Code Assistant Guidelines
## This is a windows project, so use windows commands when building depdencies and running things: python, pip, etc

## Critical Configuration Constraints

### DO NOT MODIFY - Model Configuration
The following model configurations are optimized for this project and must NOT be changed in future edits:

#### Primary LLM (backend/processors/llm.py)
- **Model Path**: `models/Phi-3-medium-4k-instruct.gguf`
- **Context Length**: `n_ctx=2048`
- **Reasoning**: This configuration is optimized for the specific use case and performance requirements

#### Alternative ExLlama LLM (backend/processors/exllama_processor.py)  
- **Model Path**: `models/Phi-3-medium-4k-instruct-exl2-4_25`
- **Max Sequence Length**: Uses model's default from config
- **Reasoning**: ExLlama alternative maintained for performance comparison

### Fixed Parameters
- Context window size: **2048 tokens** (do not increase or decrease)
- Model selection: **Phi-3-medium variants only**
- GPU acceleration: Current settings optimized for hardware

## CRITICAL REQUIREMENT - GPU-Only Execution
**NEVER ALLOW CPU MODEL EXECUTION**: All models MUST run on GPU. If GPU support is lost or models fall back to CPU execution, this is considered a critical failure that must be immediately fixed. Do not test, accept, or work around CPU execution - always restore full GPU acceleration.

## Environment Setup
**ALWAYS USE VIRTUAL ENVIRONMENT**: This project requires the virtual environment at `/home/kevin/orderly/venv/` which contains:
- Modified llama-cpp-python with GPU CUDA support
- GPU cancellation functionality for real-time STT→LLM pipeline
- All required dependencies with proper versions

**Critical Dependencies**:
- **NumPy**: Must be >= 1.22, < 2.0 (for nemo-toolkit and numba compatibility)
- **CUDA**: Version 12.9 with proper toolkit and nvcc compiler
- **llama-cpp-python**: Custom build with cancellation support

**To run any Python code**: `source /home/kevin/orderly/venv/bin/activate && cd /home/kevin/orderly/backend && python <script>`

### What CAN be modified
- Generation parameters (temperature, top_k, etc.)
- Prompt templates and system messages
- Response processing logic
- Performance monitoring and logging
- API endpoints and routing
- Audio processing components
- Frontend components

### Project Overview
This is an AI-powered order taking system with:
- Speech-to-text processing (Whisper/Parakeet)
- LLM reasoning for order processing (Phi-3)
- Text-to-speech output (Orpheus TTS)
- Real-time WebSocket communication
- React frontend interface

### Development Guidelines
When making changes:
1. Preserve existing model paths and context settings
2. Test changes without modifying core LLM configuration
3. Focus on business logic, API improvements, and user experience
4. Maintain compatibility with existing audio processing pipeline

## Audio Processing Dependency: FFmpeg

This project requires [FFmpeg](https://ffmpeg.org/) to be installed and available in your system PATH for audio processing (used by pydub and NeMo).

### Windows Installation (Recommended)

1. Open PowerShell as Administrator (right-click → "Run as administrator").
2. Install Chocolatey if you don't have it:
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```
3. Install ffmpeg:
   ```powershell
   choco install ffmpeg -y
   ```

After installation, restart your terminal and verify ffmpeg is available:
```powershell
ffmpeg -version
```
You should see version information printed.

If you install ffmpeg manually, ensure the folder containing `ffmpeg.exe` is added to your system PATH.
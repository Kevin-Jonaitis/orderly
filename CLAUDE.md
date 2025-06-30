# Claude Code Assistant Guidelines

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
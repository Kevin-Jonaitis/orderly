# Silence-Based STT System

A real-time speech-to-text system that collects audio until 200ms of silence is detected, then transcribes the complete audio segment using NeMo FastConformer.

## Features

- **Silence Detection**: Automatically detects when speech ends using RMS-based silence detection
- **Complete Audio Transcription**: Transcribes entire speech segments rather than streaming partial results
- **Timing Measurements**: Provides detailed timing information for transcription performance
- **Model Warm-up**: Includes warm-up step with test audio for consistent performance
- **Audio Buffering**: Efficiently buffers audio chunks until silence triggers transcription

## How It Works

1. **Audio Collection**: Captures 80ms audio chunks at 24kHz
2. **Silence Detection**: Monitors RMS levels to detect silence (threshold: 0.01)
3. **Audio Buffering**: Accumulates audio chunks in memory
4. **Silence Timeout**: After 200ms of continuous silence, triggers transcription
5. **Complete Transcription**: Transcribes the entire buffered audio segment
6. **Timing Analysis**: Measures and reports transcription timing

## Files

- `silence_based_stt.py` - Main silence-based STT implementation
- `test_silence_stt.py` - Test suite for silence detection and NeMo integration
- `README_silence_stt.md` - This documentation

## Usage

### Prerequisites

```bash
# Activate Python 3.12 virtual environment
source venv312/Scripts/activate  # Windows
# or
source venv312/bin/activate      # Linux/Mac

# Install dependencies
pip install nemo_toolkit[asr] sounddevice soundfile numpy
```

### Running Silence-Based STT

```bash
# List available audio devices
python silence_based_stt.py --list-devices

# Run with default microphone
python silence_based_stt.py

# Run with specific device ID
python silence_based_stt.py --device 1
```

### Testing

```bash
# Run test suite
python test_silence_stt.py
```

## Configuration

### Audio Settings
- **Input Sample Rate**: 24kHz
- **Target Sample Rate**: 16kHz (for NeMo)
- **Chunk Duration**: 80ms
- **Channels**: Mono

### Silence Detection
- **Silence Threshold**: 0.01 RMS
- **Silence Timeout**: 200ms
- **Detection Method**: RMS-based

### NeMo Model
- **Model**: `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi`
- **Latency Config**: `[70, 13]` (approx 800ms)
- **Decoder**: RNNT

## Output Format

```
🎤 Segment 1: 'Hello world this is a test'
⏱️  Transcribe time: 245.67 milliseconds
🎵 Audio length: 2.34 seconds
📊 Buffer size: 37440 samples
--------------------------------------------------
```

## Performance Considerations

- **Memory Usage**: Audio is buffered in memory until transcription
- **Latency**: 200ms silence detection + transcription time
- **Accuracy**: Complete audio segments provide better accuracy than streaming
- **Warm-up**: Model is warmed up with test_audio.wav for consistent performance

## Comparison with Streaming STT

| Aspect | Streaming STT | Silence-Based STT |
|--------|---------------|-------------------|
| Latency | Low (real-time) | Higher (200ms + transcription) |
| Accuracy | Lower (partial results) | Higher (complete segments) |
| Memory | Low | Higher (audio buffering) |
| Use Case | Real-time applications | Post-processing, analysis |

## Troubleshooting

### Common Issues

1. **NeMo Import Error**: Install with `pip install nemo_toolkit[asr]`
2. **Audio Device Issues**: Use `--list-devices` to find correct device ID
3. **Silence Detection**: Adjust `SILENCE_THRESHOLD` for your environment
4. **Performance**: Ensure test_audio.wav exists for proper warm-up

### Debug Mode

The system includes progress indicators and detailed timing information to help diagnose issues.

## Future Enhancements

- Configurable silence thresholds
- Multiple audio format support
- Batch processing mode
- Integration with other STT models
- Real-time visualization of audio levels 
"""
Audio processing utilities.

This module provides:
- convert_webm_to_wav: Convert WebM audio to WAV format for STT processing
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

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
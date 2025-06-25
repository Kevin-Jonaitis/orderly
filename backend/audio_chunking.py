"""
Audio chunking utilities for 480ms streaming STT processing.

This module handles:
- WAV file parsing and data extraction
- 480ms chunk generation with proper WAV headers
- Pre-chunked file management for streaming inference
"""

import struct
import time
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def parse_wav_data(wav_bytes: bytes) -> Optional[bytes]:
    """Extract raw audio data from WAV file, handling complex headers"""
    
    if len(wav_bytes) < 12:
        logger.error("WAV file too small")
        return None
    
    # Find the data chunk by parsing the WAV structure
    data_offset = 12  # Skip RIFF header (12 bytes: RIFF + size + WAVE)
    
    while data_offset < len(wav_bytes) - 8:
        chunk_id = wav_bytes[data_offset:data_offset + 4]
        chunk_size = struct.unpack('<I', wav_bytes[data_offset + 4:data_offset + 8])[0]
        
        if chunk_id == b'data':
            data_start = data_offset + 8
            audio_data = wav_bytes[data_start:data_start + chunk_size]
            logger.info(f"Found data chunk at offset {data_start}, size: {len(audio_data)} bytes")
            return audio_data
        else:
            # Skip this chunk (add padding if chunk size is odd)
            data_offset += 8 + chunk_size + (chunk_size % 2)
    
    logger.error("No 'data' chunk found in WAV file")
    return None


def create_wav_header(data_size: int) -> bytes:
    """Create a minimal WAV header for 16kHz, 16-bit mono audio"""
    header = bytearray(44)
    
    # RIFF header
    struct.pack_into('<4s', header, 0, b'RIFF')
    struct.pack_into('<I', header, 4, 36 + data_size)  # File size - 8
    struct.pack_into('<4s', header, 8, b'WAVE')
    
    # fmt chunk
    struct.pack_into('<4s', header, 12, b'fmt ')
    struct.pack_into('<I', header, 16, 16)  # fmt chunk size
    struct.pack_into('<H', header, 20, 1)   # PCM format
    struct.pack_into('<H', header, 22, 1)   # Mono
    struct.pack_into('<I', header, 24, 16000)  # Sample rate
    struct.pack_into('<I', header, 28, 32000)  # Byte rate (16000 * 2 * 1)
    struct.pack_into('<H', header, 32, 2)   # Block align
    struct.pack_into('<H', header, 34, 16)  # Bits per sample
    
    # data chunk header
    struct.pack_into('<4s', header, 36, b'data')
    struct.pack_into('<I', header, 40, data_size)
    
    return bytes(header)


def chunk_wav_audio(wav_bytes: bytes, chunk_duration_ms: int = 480) -> List[bytes]:
    """Split WAV audio into chunks of specified duration (in milliseconds)"""
    
    # WAV file format constants
    SAMPLE_RATE = 16000  # 16kHz
    BYTES_PER_SAMPLE = 2  # 16-bit
    CHANNELS = 1  # Mono
    
    # Calculate chunk size in bytes
    chunk_size_samples = int((chunk_duration_ms / 1000.0) * SAMPLE_RATE)
    chunk_size_bytes = chunk_size_samples * BYTES_PER_SAMPLE * CHANNELS
    
    # Extract audio data from WAV file
    audio_data = parse_wav_data(wav_bytes)
    if audio_data is None:
        return []
    
    logger.info(f"Chunk size: {chunk_size_bytes} bytes ({chunk_duration_ms}ms)")
    
    chunks = []
    offset = 0
    chunk_num = 0
    
    while offset < len(audio_data):
        # Extract chunk data
        chunk_data = audio_data[offset:offset + chunk_size_bytes]
        
        # For the last chunk, pad with silence if necessary
        if len(chunk_data) < chunk_size_bytes:
            silence_padding = b'\x00' * (chunk_size_bytes - len(chunk_data))
            chunk_data += silence_padding
            logger.info(f"Chunk {chunk_num}: Padded {len(silence_padding)} bytes of silence")
        
        # Create new WAV file with minimal header
        chunk_header = create_wav_header(len(chunk_data))
        chunk_wav = chunk_header + chunk_data
        chunks.append(chunk_wav)
        
        logger.debug(f"Chunk {chunk_num}: {len(chunk_wav)} bytes ({len(chunk_data)} audio + 44 header)")
        offset += chunk_size_bytes
        chunk_num += 1
    
    logger.info(f"Split into {len(chunks)} chunks of {chunk_duration_ms}ms each")
    return chunks


def generate_chunk_files(input_wav_path: str, output_dir: str, chunk_duration_ms: int = 480) -> bool:
    """Generate chunk files from input WAV and save to output directory"""
    
    input_path = Path(input_wav_path)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input WAV file not found: {input_path}")
        return False
    
    # Load the input audio file
    logger.info(f"Loading {input_path}...")
    with open(input_path, "rb") as f:
        wav_bytes = f.read()
    
    logger.info(f"Loaded {len(wav_bytes)} bytes")
    
    # Generate chunks
    logger.info(f"Generating {chunk_duration_ms}ms chunks...")
    chunks = chunk_wav_audio(wav_bytes, chunk_duration_ms)
    
    if not chunks:
        logger.error("Failed to generate chunks")
        return False
    
    # Clear existing chunks
    if output_path.exists():
        for existing_chunk in output_path.glob("chunk_*.wav"):
            existing_chunk.unlink()
        logger.info(f"Cleared existing chunks from {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created chunks directory: {output_path}")
    
    # Save chunks to disk
    logger.info(f"Saving {len(chunks)} chunks to {output_path}/...")
    for i, chunk_wav in enumerate(chunks):
        chunk_path = output_path / f"chunk_{i:03d}.wav"
        with open(chunk_path, "wb") as f:
            f.write(chunk_wav)
        logger.debug(f"Saved {chunk_path} ({len(chunk_wav)} bytes)")
    
    logger.info(f"Successfully generated {len(chunks)} chunks in {output_path}/")
    logger.info(f"Chunk files: chunk_000.wav through chunk_{len(chunks)-1:03d}.wav")
    return True


def get_chunk_files(chunks_dir: str) -> List[Path]:
    """Get sorted list of chunk files from directory"""
    
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        logger.error(f"Chunks directory not found: {chunks_path}")
        return []
    
    # Find all chunk files
    chunk_files = sorted(chunks_path.glob("chunk_*.wav"))
    if not chunk_files:
        logger.error(f"No chunk files found in {chunks_path}")
        return []
    
    logger.info(f"Found {len(chunk_files)} chunk files in {chunks_path}")
    return chunk_files


def estimate_audio_duration(num_chunks: int, chunk_duration_ms: int = 480) -> float:
    """Estimate total audio duration from number of chunks"""
    return (num_chunks * chunk_duration_ms) / 1000.0


class ChunkProcessor:
    """Handles processing of pre-generated audio chunks"""
    
    def __init__(self, chunks_dir: str = "test/chunks"):
        self.chunks_dir = chunks_dir
        self.chunk_files = []
        self._load_chunks()
    
    def _load_chunks(self):
        """Load chunk files from directory"""
        self.chunk_files = get_chunk_files(self.chunks_dir)
        if not self.chunk_files:
            logger.warning(f"No chunks found in {self.chunks_dir}")
            logger.warning("Run 'python3 generate_chunks.py' to create chunk files")
    
    def reload_chunks(self):
        """Reload chunk files from directory"""
        self._load_chunks()
    
    def get_chunk_count(self) -> int:
        """Get number of available chunks"""
        return len(self.chunk_files)
    
    def get_estimated_duration(self) -> float:
        """Get estimated total audio duration in seconds"""
        return estimate_audio_duration(len(self.chunk_files))
    
    def iter_chunk_paths(self):
        """Iterate over chunk file paths"""
        for chunk_file in self.chunk_files:
            yield str(chunk_file)
    
    def is_ready(self) -> bool:
        """Check if chunks are available for processing"""
        return len(self.chunk_files) > 0
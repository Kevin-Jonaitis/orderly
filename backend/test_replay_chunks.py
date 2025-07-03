#!/usr/bin/env python3
"""
Test script to replay saved audio chunks directly to the audio queue
This bypasses TTS generation for testing the audio pipeline
"""

import pickle
import sounddevice as sd
import queue
import time
import sys
import numpy as np

def replay_chunks(chunk_file):
    """Load and replay saved audio chunks"""
    print(f"🎵 Loading chunks from {chunk_file}...")
    
    # Load the chunks
    chunks = pickle.load(open(chunk_file, 'rb'))
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # Set up audio queue and playback
    audio_queue = queue.Queue()
    
    def audio_callback(outdata, frames, time, status):
        """Simple audio callback"""
        try:
            audio_chunk = audio_queue.get_nowait()
            # Handle both 2048-sample chunks and 512-sample blocks
            if len(audio_chunk.shape) > 1:
                audio_chunk = audio_chunk.flatten()
            
            # Ensure we have the right amount of data
            if len(audio_chunk) >= frames:
                outdata[:, 0] = audio_chunk[:frames]
            else:
                # Pad with zeros if needed
                outdata[:, 0] = np.pad(audio_chunk, (0, frames - len(audio_chunk)))
        except queue.Empty:
            outdata.fill(0)
    
    # Create output stream with 512-sample blocks (matching TTS process)
    stream = sd.OutputStream(
        callback=audio_callback,
        samplerate=24000,
        channels=1,
        dtype='float32',
        blocksize=512,
        latency=0.002
    )
    
    print(f"🔊 Stream config: blocksize={stream.blocksize}, latency={stream.latency}")
    
    # Start playback
    stream.start()
    print("▶️  Starting playback...")
    
    # Queue all chunks with same 512-sample splitting as TTS
    for i, audio_array in enumerate(chunks):
        # Flatten the array
        audio_flat = audio_array.flatten()
        
        # Split into 512-sample blocks (same as TTS process)
        chunk_size = 512
        num_blocks = len(audio_flat) // chunk_size
        
        for j in range(num_blocks):
            start_idx = j * chunk_size
            end_idx = start_idx + chunk_size
            audio_block = audio_flat[start_idx:end_idx]
            audio_queue.put(audio_block)
        
        # Handle remaining samples
        remaining_samples = len(audio_flat) % chunk_size
        if remaining_samples > 0:
            remaining_audio = audio_flat[num_blocks * chunk_size:]
            padded_audio = np.pad(remaining_audio, (0, chunk_size - remaining_samples), mode='constant')
            audio_queue.put(padded_audio)
        
        print(f"📦 Queued chunk {i+1}/{len(chunks)}")
    
    # Wait for playback to complete
    print("🎵 Playing audio...")
    while not audio_queue.empty():
        time.sleep(0.1)
    
    # Extra time for last chunks to play
    time.sleep(0.5)
    
    # Cleanup
    stream.stop()
    stream.close()
    print("✅ Playback complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_replay_chunks.py <chunk_file.pkl>")
        print("Example: python test_replay_chunks.py debug_chunks_1701234567.pkl")
        sys.exit(1)
    
    replay_chunks(sys.argv[1])
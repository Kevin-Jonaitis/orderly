#!/usr/bin/env python3
"""
Generate beep audio chunks for testing AudioProcessor latency
Creates a simple beep tone and splits it into chunks for testing
"""

import numpy as np
import pickle
import time

def generate_beep(frequency=1000, duration=2.0, sample_rate=24000):
    """Generate a beep tone (based on test_wsl_audio_latency.py)"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

def split_into_chunks(audio_data, chunk_size=2048):
    """Split audio data into chunks for streaming"""
    chunks = []
    audio_flat = audio_data.flatten()
    
    # Split into blocks
    num_blocks = len(audio_flat) // chunk_size
    for i in range(num_blocks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk = audio_flat[start_idx:end_idx]
        chunks.append(chunk)
    
    # Handle remaining samples
    remaining_samples = len(audio_flat) % chunk_size
    if remaining_samples > 0:
        remaining_audio = audio_flat[num_blocks * chunk_size:]
        padded_audio = np.pad(remaining_audio, (0, chunk_size - remaining_samples), mode='constant')
        chunks.append(padded_audio)
    
    return chunks

def main():
    print("🎵 Generating beep chunks for AudioProcessor testing...")
    
    # Generate a clear beep tone
    print("🔧 Generating 2-second beep at 1000Hz...")
    beep_audio = generate_beep(frequency=1000, duration=2.0, sample_rate=24000)
    
    # Split into chunks
    print("🔧 Splitting into 2048-sample chunks...")
    chunks = split_into_chunks(beep_audio, chunk_size=2048)
    
    print(f"✅ Generated {len(chunks)} chunks")
    print(f"   Audio duration: 2.0 seconds")
    print(f"   Sample rate: 24000 Hz")
    print(f"   Chunk size: 2048 samples")
    print(f"   Total samples: {len(beep_audio)}")
    
    # Save chunks to file
    timestamp = int(time.time())
    filename = f"debug_beep_chunks_{timestamp}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"💾 Saved chunks to: {filename}")
    
    # Verify by loading
    with open(filename, 'rb') as f:
        loaded_chunks = pickle.load(f)
    
    print(f"✅ Verification: loaded {len(loaded_chunks)} chunks from file")
    
    return filename

if __name__ == "__main__":
    filename = main()
    print(f"\n🎯 Use this file in your tests: {filename}")
    print("   Update debug chunk loading in test_audio_processor.py and tts_process.py")
#!/usr/bin/env python3
"""
Test the optimized ParakeetSTTProcessor to compare performance with nemo_streaming_test.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ParakeetSTTProcessor
import time
import asyncio

async def test_optimized_stt():
    print("🧪 Testing optimized ParakeetSTTProcessor...")
    
    # Initialize processor
    print("🚀 Initializing processor...")
    start_init = time.time()
    processor = ParakeetSTTProcessor()
    init_time = (time.time() - start_init) * 1000
    print(f"⏱️  Initialization took: {init_time:.0f}ms")
    
    # Test transcription
    print("🎯 Testing transcription...")
    start_total = time.time()
    
    # Load test audio file
    with open("test/test_audio.wav", "rb") as f:
        wav_bytes = f.read()
    
    # Transcribe
    result = await processor.transcribe(wav_bytes)
    
    total_time = (time.time() - start_total) * 1000
    print(f"🎤 Final result: '{result}'")
    print(f"⏱️  Total time: {total_time:.0f}ms")
    
    return result, total_time

if __name__ == "__main__":
    asyncio.run(test_optimized_stt())
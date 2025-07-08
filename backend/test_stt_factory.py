#!/usr/bin/env python3
"""
Test script to verify STT factory and processors work correctly with timing
"""

import sys
import os
import asyncio
import time
import tempfile

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processors.stt import create_stt_processor

def check_gpu_status():
    """Check GPU availability and status"""
    print("🔍 Checking GPU status...")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"   CUDA devices: {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   Device {i}: {device_name} ({memory_total:.1f} GB)")
            
            # Check current device
            current_device = torch.cuda.current_device()
            print(f"   Current CUDA device: {current_device}")
        else:
            print("   ⚠️  CUDA not available - models will run on CPU")
            
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    # Check Faster Whisper GPU support
    try:
        import faster_whisper
        print(f"   Faster Whisper available: ✅")
    except ImportError:
        print("   ❌ Faster Whisper not installed")

def read_wav_file(file_path: str) -> bytes:
    """Read WAV file and return its bytes"""
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None

async def test_processors():
    """Test both STT processors with warm-up and timing"""
    print("🧪 Testing STT Factory and Processors with Timing")
    print("=" * 60)
    
    # Check GPU status first
    check_gpu_status()
    
    # Test files
    warmup_file = "test_audio.wav"
    test_file = "test_aws_speech.wav"
    
    # Test NeMo processor
    print("\n1. Testing NeMo Processor...")
    try:
        nemo_processor = create_stt_processor("nemo")
        print("✅ NeMo processor created successfully")
        
        # Check if model is on GPU
        try:
            import torch
            if hasattr(nemo_processor, 'model'):
                device = next(nemo_processor.model.parameters()).device
                print(f"   NeMo model device: {device}")
                if device.type == 'cuda':
                    print(f"   ✅ NeMo model is on GPU: {device}")
                else:
                    print(f"   ⚠️  NeMo model is on CPU: {device}")
        except Exception as e:
            print(f"   ⚠️  Could not check NeMo model device: {e}")
        
        # Test reset_state method
        nemo_processor.reset_state()
        print("✅ NeMo reset_state() works")
        
        # Warm up with test_audio.wav using direct file transcription (like silence_based_stt.py)
        if os.path.exists(warmup_file):
            print(f"🔥 Warming up NeMo with {warmup_file}...")
            warmup_start = time.time()
            # Use direct file transcription like silence_based_stt.py
            warmup_result = nemo_processor.model.transcribe([warmup_file])
            warmup_elapsed = (time.time() - warmup_start) * 1000
            print(f"🔥 NeMo warm-up transcription: {warmup_result[0].text}")
            print(f"🔥 NeMo warm-up time: {warmup_elapsed:.2f} milliseconds")
        else:
            print(f"⚠️  {warmup_file} not found, skipping NeMo warm-up")
        
        # Test transcription with test_aws_speech.wav using direct file transcription
        if os.path.exists(test_file):
            print(f"\n Transcribing {test_file} with NeMo...")
            transcribe_start = time.time()
            # Use direct file transcription like silence_based_stt.py
            result = nemo_processor.model.transcribe([test_file])
            transcribe_elapsed = (time.time() - transcribe_start) * 1000
            print(f" NeMo transcription: {result[0].text}")
            print(f" NeMo transcription time: {transcribe_elapsed:.2f} milliseconds")
        else:
            print(f"❌ {test_file} not found, skipping NeMo transcription")
        
    except Exception as e:
        print(f"❌ NeMo processor failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Faster Whisper processor
    print("\n2. Testing Faster Whisper Processor...")
    try:
        fw_processor = create_stt_processor("faster-whisper", model_size="tiny", device="cuda", compute_type="float16")
        print("✅ Faster Whisper processor created successfully")
        
        # Check if model is on GPU
        try:
            if hasattr(fw_processor, 'model'):
                device = fw_processor.model.device
                print(f"   Faster Whisper model device: {device}")
                if "cuda" in str(device).lower():
                    print(f"   ✅ Faster Whisper model is on GPU: {device}")
                else:
                    print(f"   ⚠️  Faster Whisper model is on CPU: {device}")
        except Exception as e:
            print(f"   ⚠️  Could not check Faster Whisper model device: {e}")
        
        # Test reset_state method
        fw_processor.reset_state()
        print("✅ Faster Whisper reset_state() works")
        
        # Warm up with test_audio.wav
        if os.path.exists(warmup_file):
            print(f"🔥 Warming up Faster Whisper with {warmup_file}...")
            warmup_start = time.time()
            warmup_result = await fw_processor.transcribe(read_wav_file(warmup_file))
            warmup_elapsed = (time.time() - warmup_start) * 1000
            print(f"🔥 Faster Whisper warm-up transcription: {warmup_result}")
            print(f"🔥 Faster Whisper warm-up time: {warmup_elapsed:.2f} milliseconds")
        else:
            print(f"⚠️  {warmup_file} not found, skipping Faster Whisper warm-up")
        
        # Test transcription with test_aws_speech.wav
        if os.path.exists(test_file):
            print(f"\n Transcribing {test_file} with Faster Whisper...")
            transcribe_start = time.time()
            result = await fw_processor.transcribe(read_wav_file(test_file))
            transcribe_elapsed = (time.time() - transcribe_start) * 1000
            print(f"📝 Faster Whisper transcription: {result}")
            print(f"📝 Faster Whisper transcription time: {transcribe_elapsed:.2f} milliseconds")
        else:
            print(f"❌ {test_file} not found, skipping Faster Whisper transcription")
        
    except Exception as e:
        print(f"❌ Faster Whisper processor failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Factory test with timing completed!")

if __name__ == "__main__":
    asyncio.run(test_processors()) 
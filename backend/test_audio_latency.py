#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import time

def test_audio_latency():
    """Test raw audio output latency with a simple beep"""
    
    # Generate a 1000Hz beep for 0.5 seconds
    duration = 0.5  # seconds
    sample_rate = 24000  # Match your TTS sample rate
    frequency = 1000  # Hz
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    beep = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    print("🔊 Audio Latency Test")
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Audio device info:")
    print(sd.query_devices())
    print()
    
    # Test with default settings (like your current TTS)
    print("📊 Testing DEFAULT sounddevice settings...")
    print("Press Enter, then immediately start timing when you hear the beep:")
    input()
    
    start_time = time.time()
    sd.play(beep, samplerate=sample_rate)
    sd.wait()  # Wait until playback is finished
    
    print(f"⏱️  Beep should have started at: {start_time:.3f}")
    print("When did you actually hear it? (Compare timestamps)")
    print()
    
    # Test with low-latency settings
    print("📊 Testing LOW-LATENCY settings...")
    print("Press Enter for second test:")
    input()
    
    start_time = time.time()
    sd.play(beep, samplerate=sample_rate, blocksize=64, latency='low')
    sd.wait()
    
    print(f"⏱️  Low-latency beep started at: {start_time:.3f}")
    print("When did you hear this one?")
    print()
    
    # Test with 48kHz (match your system)
    print("📊 Testing 48kHz (matching your system)...")
    print("Press Enter for third test:")
    input()
    
    beep_48k = np.interp(np.linspace(0, len(beep), int(len(beep) * 48000/24000)), 
                         np.arange(len(beep)), beep).astype(np.float32)
    
    start_time = time.time()
    sd.play(beep_48k, samplerate=48000, blocksize=64, latency='low')
    sd.wait()
    
    print(f"⏱️  48kHz beep started at: {start_time:.3f}")
    print("When did you hear this one?")

if __name__ == "__main__":
    test_audio_latency()
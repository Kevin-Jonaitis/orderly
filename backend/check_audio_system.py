#!/usr/bin/env python3
"""
Audio System Configuration Checker
Check WSL2, ALSA, PulseAudio configuration for latency issues
"""

import subprocess
import sounddevice as sd
import os

def run_command(cmd, description):
    """Run a system command and display results"""
    print(f"\n🔍 {description}")
    print("-" * len(description))
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"Error running command: {e}")

def check_audio_system():
    """Check various audio system components"""
    
    print("🔧 AUDIO SYSTEM DIAGNOSTIC")
    print("=" * 40)
    
    # Check if we're in WSL
    run_command("uname -a", "System Information")
    
    # Check audio devices
    print("\n🎵 SoundDevice Info:")
    print("-" * 20)
    try:
        print("Default input device:", sd.default.device[0])
        print("Default output device:", sd.default.device[1])
        print("Default sample rate:", sd.default.samplerate)
        print("Default channels:", sd.default.channels)
        print("Default dtype:", sd.default.dtype)
        print("\nAll devices:")
        print(sd.query_devices())
    except Exception as e:
        print(f"SoundDevice error: {e}")
    
    # Check ALSA
    run_command("aplay -l", "ALSA Playback Devices")
    run_command("aplay -L", "ALSA Device Names")
    
    # Check PulseAudio
    run_command("pulseaudio --check", "PulseAudio Status")
    run_command("pactl info", "PulseAudio Info")
    run_command("pactl list short sinks", "PulseAudio Sinks")
    run_command("pactl list short sources", "PulseAudio Sources")
    
    # Check PulseAudio latency settings
    run_command("pactl list sinks | grep -A 10 'Latency\\|Buffer'", "PulseAudio Latency Settings")
    
    # Check environment variables
    print("\n🌍 Audio Environment Variables:")
    print("-" * 30)
    audio_vars = ['PULSE_RUNTIME_PATH', 'PULSE_RUNTIME_DIR', 'PULSE_SERVER', 
                  'ALSA_CARD', 'ALSA_DEVICE', 'SDL_AUDIODRIVER']
    for var in audio_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    # Check WSL-specific audio setup
    run_command("ls -la /mnt/c/Windows/System32/", "Windows System Access (WSL check)")
    
    # Check audio processes
    run_command("ps aux | grep -E '(pulse|alsa|audio)'", "Audio Processes")
    
    # Check kernel audio modules
    run_command("lsmod | grep snd", "Sound Kernel Modules")
    
    # Check /proc/asound for detailed ALSA info
    run_command("cat /proc/asound/cards", "ALSA Sound Cards")
    run_command("cat /proc/asound/version", "ALSA Version")

if __name__ == "__main__":
    check_audio_system()
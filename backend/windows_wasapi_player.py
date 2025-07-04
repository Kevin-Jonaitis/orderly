#!/usr/bin/env python3
"""
Windows DirectSound Low-Latency Audio Player
Uses DirectSound for low-latency audio playback on Windows.
Reads audio chunks from multiprocessing.Queue for integration with TTS pipeline.
"""

import time
import queue
import threading
import multiprocessing
import numpy as np
import sounddevice as sd
from typing import Optional


class WindowsWASAPIPlayer:
    """Low-latency audio player using Windows DirectSound.
    
    Designed for <20ms latency audio playback from multiprocessing.Queue.
    """
    
    def __init__(self, audio_queue: multiprocessing.Queue, 
                 sample_rate: int = 48000, 
                 channels: int = 1,
                 chunk_size: int = 2048,
                 target_latency_ms: float = 5.0,
                 device_id: Optional[int] = None):
        """Initialize WASAPI player.
        
        Args:
            audio_queue: multiprocessing.Queue containing audio chunks (numpy arrays)
            sample_rate: Audio sample rate (match your hardware, typically 48000)
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Audio buffer size in samples (default 2048 for TTS chunks)
            target_latency_ms: Target latency in milliseconds
        """
        self.audio_queue = audio_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.target_latency_ms = target_latency_ms
        self.device_id = device_id
        
        # Stream state
        self.stream = None
        self.running = False
        
        # Timing analysis
        self.first_chunk_received = None
        self.first_write_time = None
        self.chunks_written = 0
        
        print(f"🎵 Windows DirectSound Player initialized:")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Channels: {channels}")
        print(f"   Chunk size: {chunk_size} samples")
        print(f"   Target latency: {target_latency_ms}ms")
    
    def _setup_wasapi_stream(self):
        """Setup DirectSound audio stream."""
        return self._setup_directsound()
    
    def _setup_directsound(self):
        """Try to setup DirectSound stream."""
        try:
            # Find DirectSound host API
            ds_hostapi = None
            for i, hostapi in enumerate(sd.query_hostapis()):
                if 'DirectSound' in hostapi['name']:
                    ds_hostapi = i
                    break
            
            if ds_hostapi is None:
                print("⚠️ DirectSound host API not found")
                return False
            
            print(f"🔧 Using DirectSound host API: {sd.query_hostapis()[ds_hostapi]['name']}")
            
            # Get output device
            devices = sd.query_devices()
            target_device = self._get_target_device(devices)
            
            device_info = devices[target_device]
            print(f"🔧 Using device: {device_info['name']}")
            
            # Set device as default
            sd.default.device[1] = target_device
            
            # Create DirectSound stream for blocking writes (no callback)
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                latency=self.target_latency_ms / 1000.0
            )
            
            print(f"✅ DirectSound stream created - latency: {self.stream.latency*1000:.2f}ms")
            return True
            
        except Exception as e:
            print(f"❌ DirectSound failed: {e}")
            return False
    
    def _get_target_device(self, devices):
        """Get the target output device."""
        if self.device_id is not None:
            if (self.device_id < len(devices) and 
                devices[self.device_id]['max_output_channels'] > 0):
                return self.device_id
            else:
                raise RuntimeError(f"Device {self.device_id} is not a valid output device")
        else:
            # Use system default output device
            default_device = sd.default.device[1]
            if default_device is None or default_device == -1:
                # Find first available output
                for i, device in enumerate(devices):
                    if device['max_output_channels'] > 0:
                        return i
            else:
                return default_device
            
            raise RuntimeError("No output device found")
    
    def _blocking_write_loop(self):
        """Simple blocking write loop - reads from queue and writes to stream."""
        print("🎵 Starting blocking audio write loop...")
        
        while self.running:
            try:
                # Get audio chunk from multiprocessing queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                if audio_chunk is None:
                    print("🛑 Received shutdown signal")
                    break
                
                current_time = time.time()
                
                # First chunk timing
                if self.first_chunk_received is None:
                    self.first_chunk_received = current_time
                    print(f"📦 FIRST CHUNK RECEIVED: {current_time:.6f}")
               
                # Record first write timing
                if self.first_write_time is None:
                    self.first_write_time = time.time()
                    print(f"🔊 FIRST WRITE: {self.first_write_time:.6f}")
                    print(f"📊 Queue-to-write latency: {(self.first_write_time - self.first_chunk_received)*1000:.2f}ms")
                
                # Direct blocking write to stream
                self.stream.write(audio_chunk)
                self.chunks_written += 1
                
            except queue.Empty:
                # Timeout is normal - just continue
                continue
            except Exception as e:
                print(f"❌ Error in blocking write loop: {e}")
                self.running = False
                break
        
        print("🛑 Blocking write loop finished")
    
    def start(self):
        """Start the audio player."""
        print("🚀 Starting Windows DirectSound audio player...")
        
        # Setup DirectSound stream
        if not self._setup_directsound():
            return False
        
        # Start audio stream
        try:
            self.stream.start()
            print("✅ DirectSound stream started - ready for audio chunks")
            
            # Set running flag and start blocking write loop
            self.running = True
            self._blocking_write_loop()
            
            return True
        except Exception as e:
            print(f"❌ Failed to start DirectSound stream: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop the audio player and cleanup."""
        print("🛑 Stopping Windows DirectSound audio player...")
        
        # Stop running flag
        self.running = False
        
        # Stop audio stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                print("✅ DirectSound stream stopped")
            except Exception as e:
                print(f"⚠️ Error stopping stream: {e}")
        
        self._print_timing_stats()
    
    def _print_timing_stats(self):
        """Print timing analysis."""
        if not self.first_chunk_received or not self.first_write_time:
            print("📊 No timing data available")
            return
        
        print("\n📊 TIMING ANALYSIS:")
        print(f"   First chunk received: {self.first_chunk_received:.6f}")
        print(f"   First write time: {self.first_write_time:.6f}")
        print(f"   Queue-to-write latency: {(self.first_write_time - self.first_chunk_received)*1000:.2f}ms")
        print(f"   Total chunks written: {self.chunks_written}")


def test_directsound_player():
    """Test function to verify DirectSound player works."""
    print("🧪 Testing Windows DirectSound Player...")
    
    # Use system default audio device
    print("🔧 Using system default audio device")
    
    # Manual timing variables
    first_chunk_time = None
    audio_heard_time = None
    
    # Create a test queue and load pre-recorded chunks
    test_queue = multiprocessing.Queue()
    
    # Load pre-recorded audio chunks
    try:
        import pickle
        chunks = pickle.load(open('debug_beep_chunks_1751591457.pkl', 'rb'))
        print(f"📦 Loaded {len(chunks)} pre-recorded audio chunks")
        
        # Add chunks to queue (already float32 numpy arrays)
        for i, chunk in enumerate(chunks):
            if i == 0:
                first_chunk_time = time.time()
                print(f"🎵 FIRST CHUNK CREATED: {first_chunk_time:.6f}")
            test_queue.put(chunk)
            print(f"📦 Added chunk {i+1}: {len(chunk)} samples")
        
        sample_rate = 24000  # Assume this matches your chunks
        
    except Exception as e:
        print(f"❌ Failed to load debug chunks: {e}")
        print("❌ Test cannot continue without audio chunks")
        return
    
    # Add shutdown signal
    test_queue.put(None)
    
    # Create and start player
    player = WindowsWASAPIPlayer(
        audio_queue=test_queue,
        sample_rate=sample_rate,
        channels=1,
        chunk_size=2048,  # Standard TTS chunk size
        target_latency_ms=5.0
    )
    
    # Start player in a separate thread (since blocking mode blocks)
    import threading
    player_thread = threading.Thread(target=player.start)
    player_thread.start()
    
    # Wait for user to hear audio
    print("\n🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
    print("🔊 Press Enter when you HEAR the audio 🔊")
    print("🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊🔊")
    input()
    audio_heard_time = time.time()
    
    # Calculate and display manual timing
    if first_chunk_time and audio_heard_time:
        total_latency = (audio_heard_time - first_chunk_time) * 1000
        print(f"\n⚡ TOTAL END-TO-END LATENCY: {total_latency:.1f}ms")
        print(f"📊 First chunk created: {first_chunk_time:.6f}")
        print(f"📊 Audio heard: {audio_heard_time:.6f}")
    
    # Let remaining audio play
    time.sleep(2.0)
    
    # Stop player
    player.stop()
    player_thread.join(timeout=2.0)
    
    print("\n✅ Test completed")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_directsound_player()
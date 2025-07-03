#!/usr/bin/env python3
"""
WSL2-specific audio latency test
Tests various configurations to find optimal low-latency settings
"""

import sounddevice as sd
import numpy as np
import time
import os
import subprocess
import queue
import threading

def get_pulse_info():
    """Get PulseAudio configuration info"""
    print("\n🔊 PulseAudio Configuration:")
    print(f"PULSE_LATENCY_MSEC: {os.environ.get('PULSE_LATENCY_MSEC', 'Not set')}")
    
    try:
        # Get PulseAudio server info
        result = subprocess.run(['pactl', 'info'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Server Name' in line or 'Default Sink' in line or 'Default Sample' in line:
                    print(f"  {line.strip()}")
    except:
        print("  Could not get PulseAudio info")

def generate_beep(frequency=1000, duration=0.1, sample_rate=24000):
    """Generate a beep tone"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

# def test_play_latency():
#     """Test basic sd.play() latency"""
#     print("\n📊 TEST 1: Basic sd.play() latency")
#     print("=" * 50)
    
#     beep = generate_beep()
    
#     print("Press Enter, then immediately note when you hear the beep:")
#     input()
    
#     start_time = time.time()
#     print(f"🎵 PLAY CALLED: {start_time:.3f}")
#     sd.play(beep, samplerate=24000)
#     sd.wait()
    
#     print("How long after PLAY CALLED did you hear it?")

def test_callback_latency():
    """Test callback-based streaming (like your TTS)"""
    print("\n📊 TEST 3: Callback-based streaming (TTS-style)")
    print("=" * 50)
    
    audio_queue = queue.Queue()
    first_callback_time = None
    callback_count = 0
    queue_timestamps = []  # Track when chunks are queued
    dequeue_timestamps = []  # Track when chunks are dequeued
    
    def audio_callback(outdata, frames, time_info, status):
        nonlocal first_callback_time, callback_count
        
        if status:
            print(f"⚠️  Status: {status}")
        
        callback_count += 1
        current_time = time.time()
        
        if first_callback_time is None:
            first_callback_time = current_time
            print(f"🔊 FIRST CALLBACK: {current_time:.3f}")
            print(f"🔊 Requests {frames} frames")
        
        try:
            chunk = audio_queue.get_nowait()
            # time.sleep(1)
            dequeue_timestamps.append(current_time)
            outdata[:, 0] = chunk[:frames]  # Handle size mismatch
            print(f"🔊 Callback {callback_count}: consumed chunk at {current_time:.3f}")
        except queue.Empty:
            print(f"🔇 Callback {callback_count}: queue empty at {current_time:.3f}")
            # outdata.fill(0)  # Silence

        print("THE array IS: ", outdata)

    configs = [
        {'blocksize': 4, 'latency': 0.001},
        # {'blocksize': 128, 'latency': 0.001},
        # {'blocksize': 512, 'latency': 0.005},
        # {'blocksize': 0, 'latency': 'low'},
    ]
    
    beep = generate_beep(frequency=1500, duration=0.5)  # Different frequency
    
    for config in configs:
        print(f"\n🔧 Testing callback: blocksize={config['blocksize']}, latency={config['latency']}")
        input("Press Enter to test...")
        
        # Reset globals
        first_callback_time = None
        callback_count = 0
        queue_timestamps = []
        dequeue_timestamps = []
        
        # Clear queue
        while not audio_queue.empty():
            audio_queue.get()
        
        start_time = time.time()
        print(f"⏰ Start: {start_time:.3f}")
        
        try:
            stream = sd.OutputStream(
                callback=audio_callback,
                samplerate=24000,
                channels=1,
                dtype='float32',
                **config
            )
            
            print(f"📊 Actual: blocksize={stream.blocksize}, latency={stream.latency}")
            
            # Prepare chunks
            chunk_size = stream.blocksize if stream.blocksize > 0 else 512
            chunks = []
            print("THE CHUNK SIZE IS: ", chunk_size)
            for i in range(0, len(beep), chunk_size):
                chunk = beep[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunks.append(chunk)
            
            # Simplified producer thread
            stop_producer = threading.Event()
            
            def producer():
                """Simple producer: queue one chunk at a time with delay"""
                print("PUTTING THE CHUNK IN THE QUEUE")
                audio_queue.put(chunks[0])
                # for i, chunk in enumerate(chunks):
                #     if stop_producer.is_set():
                #         break
                        
                #     # Queue the chunk
                #     audio_queue.put(chunk)
                #     queue_time = time.time()
                #     queue_timestamps.append(queue_time)
                    
                #     print(f"📦 Queued chunk {i+1}/{len(chunks)} at {queue_time:.3f}")
                    
                #     # Sleep before next chunk (except after last chunk)
                #     if i < len(chunks) - 1:
                #         time.sleep(1)
                
                print(f"📦 Producer finished: queued {len(chunks)} chunks")
            
            # Start the stream first
            stream.start()
            stream_started = time.time()
            print(f"⏰ Stream started: {stream_started:.3f} (+{(stream_started-start_time)*1000:.1f}ms)")
            
            # Start producer thread
            producer_thread = threading.Thread(target=producer)
            producer_thread.start()
            
            # Wait for playback to complete
            producer_thread.join(timeout=2.0)
            stop_producer.set()
            
            # Give callbacks time to process remaining chunks
            time.sleep(2)
            
            stream.stop()
            stream.close()
            
            print(f"📊 Total callbacks: {callback_count}")
            if first_callback_time:
                print(f"📊 Time to first callback: {(first_callback_time - stream_started)*1000:.1f}ms")
            
            # Analyze timing
            if queue_timestamps and dequeue_timestamps:
                print(f"\n📊 Timing Analysis:")
                print(f"  Chunks queued: {len(queue_timestamps)}")
                print(f"  Chunks dequeued: {len(dequeue_timestamps)}")
                
                # Queue timing intervals
                if len(queue_timestamps) > 1:
                    queue_intervals = [
                        (queue_timestamps[i+1] - queue_timestamps[i]) * 1000 
                        for i in range(len(queue_timestamps)-1)
                    ]
                    print(f"  Avg queue interval: {np.mean(queue_intervals):.1f}ms (target: 20ms)")
                    print(f"  Queue interval std dev: {np.std(queue_intervals):.1f}ms")
                
                # Latency from queue to dequeue
                latencies = []
                for i in range(min(len(queue_timestamps), len(dequeue_timestamps))):
                    latencies.append((dequeue_timestamps[i] - queue_timestamps[i]) * 1000)
                
                if latencies:
                    print(f"  Avg queue→dequeue latency: {np.mean(latencies):.1f}ms")
                    print(f"  Min latency: {np.min(latencies):.1f}ms")
                    print(f"  Max latency: {np.max(latencies):.1f}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# def test_pulse_latency_env():
#     """Test with different PULSE_LATENCY_MSEC values"""
#     print("\n📊 TEST 4: PULSE_LATENCY_MSEC environment variable")
#     print("=" * 50)
    
#     beep = generate_beep()
    
#     for latency_ms in [1, 5, 10, 20, 50]:
#         print(f"\n🔧 Testing PULSE_LATENCY_MSEC={latency_ms}")
#         os.environ['PULSE_LATENCY_MSEC'] = str(latency_ms)
        
#         # Restart audio to apply setting
#         sd._terminate()
#         sd._initialize()
        
#         input("Press Enter to test...")
        
#         start_time = time.time()
#         print(f"⏰ Play called: {start_time:.3f}")
        
#         sd.play(beep, samplerate=24000, blocksize=64, latency='low')
#         sd.wait()
        
#         print("When did you hear it?")

def main():
    print("🎵 WSL2 Audio Latency Test Suite")
    print("=" * 50)
    
    # Show system info
    print("\n📊 System Information:")
    print(f"Default device: {sd.default.device}")
    print(f"Host API: {sd.query_hostapis()[0]['name']}")
    
    get_pulse_info()
    
    # Show all devices
    print("\n📊 Available Audio Devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            print(f"  [{i}] {dev['name']} - {dev['max_output_channels']}ch @ {dev['default_samplerate']}Hz")
            print(f"      Low latency: {dev['default_low_output_latency']*1000:.1f}ms")
            print(f"      High latency: {dev['default_high_output_latency']*1000:.1f}ms")
    
    print("\n🎯 Starting tests...")
    print("Listen carefully and note timing!")
    
    # Run tests
    # test_play_latency()
    # test_stream_latency()
    test_callback_latency()
    # test_pulse_latency_env()
    
    print("\n✅ Testing complete!")
    print("\nRecommendations based on your results:")
    print("- Use the configuration with lowest perceived latency")
    print("- Consider setting PULSE_LATENCY_MSEC permanently")
    print("- Check if exclusive mode or WASAPI is available")

if __name__ == "__main__":
    main()
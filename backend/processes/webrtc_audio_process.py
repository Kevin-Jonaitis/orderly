#!/usr/bin/env python3
"""
WebRTC Audio Process

Receives audio from WebRTC connections and feeds it to STT processing.
Replaces microphone capture with browser-based audio streaming.
"""

import time as time_module
import multiprocessing
from multiprocessing import Process, Queue
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)


class WebRTCAudioProcess(Process):
    """
    Process that receives audio from WebRTC connections and feeds to STT.
    
    This replaces the microphone capture functionality in STTAudioProcess
    with browser-based WebRTC audio streaming.
    """
    
    def __init__(self, stt_text_queue, webrtc_audio_queue, last_text_change_timestamp):
        super().__init__(name="WebRTCAudioProcess")
        self.stt_text_queue = stt_text_queue  # Queue to send transcribed text
        self.webrtc_audio_queue = webrtc_audio_queue  # Queue to receive WebRTC audio
        self.last_text_change_timestamp = last_text_change_timestamp
        
        # Audio processing parameters
        self.sample_rate = 24000  # Target sample rate for STT
        self.chunk_duration_ms = 100  # Process audio in 100ms chunks
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration_ms / 1000)
        
        # Audio buffering
        self.audio_buffer = np.array([], dtype=np.float32)
        self.silence_threshold = 0.01  # Adjust based on testing
        self.min_speech_duration_ms = 500  # Minimum speech length to process
        
        # Debugging
        self.frame_count = 0
        self.first_audio_time = None
        
    def run(self):
        """Main WebRTC audio processing loop"""
        logger.info("🎤 [WebRTC Audio] Starting WebRTC audio processing...")
        
        try:
            self._audio_processing_loop()
        except KeyboardInterrupt:
            logger.info("🛑 [WebRTC Audio] Keyboard interrupt - process will exit")
        except Exception as e:
            logger.error(f"❌ [WebRTC Audio] Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("🛑 [WebRTC Audio] Process complete")
    
    def _audio_processing_loop(self):
        """Main loop to process WebRTC audio frames"""
        logger.info("🎧 [WebRTC Audio] Ready - waiting for WebRTC audio...")
        
        # Initialize STT processor (similar to STTAudioProcess)
        from processors.stt import create_stt_processor
        stt_processor = create_stt_processor("realtime")
        logger.info("✅ [WebRTC Audio] STT processor initialized")
        
        while True:
            try:
                # Get audio data from WebRTC queue with timeout
                audio_data = self.webrtc_audio_queue.get(timeout=1.0)
                
                if self.first_audio_time is None:
                    self.first_audio_time = time_module.time()
                    logger.info(f"🎤 [WebRTC Audio] First audio frame at {self.first_audio_time:.3f}")
                
                self.frame_count += 1
                
                # Process the audio frame
                self._process_audio_frame(audio_data, stt_processor)
                
            except queue.Empty:
                # No audio available - continue waiting
                continue
            except Exception as e:
                logger.error(f"❌ [WebRTC Audio] Error processing frame: {e}")
                continue
    
    def _process_audio_frame(self, audio_data, stt_processor):
        """Process a single audio frame from WebRTC"""
        try:
            # Ensure audio_data is numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure float32 format
            audio_data = audio_data.astype(np.float32)
            
            # Add to buffer
            self.audio_buffer = np.append(self.audio_buffer, audio_data)
            
            # Log first few frames for debugging
            if self.frame_count <= 5:
                max_amplitude = np.max(np.abs(audio_data))
                logger.info(
                    f"🎤 [WebRTC Audio] Frame {self.frame_count}: "
                    f"{len(audio_data)} samples, max: {max_amplitude:.4f}, "
                    f"buffer: {len(self.audio_buffer)} samples"
                )
            
            # Process when we have enough audio
            while len(self.audio_buffer) >= self.samples_per_chunk:
                # Extract chunk for processing
                chunk = self.audio_buffer[:self.samples_per_chunk]
                self.audio_buffer = self.audio_buffer[self.samples_per_chunk:]
                
                # Check if chunk contains speech
                if self._is_speech(chunk):
                    # Process with STT
                    asyncio_result = self._transcribe_chunk(chunk, stt_processor)
                    
                    if asyncio_result and asyncio_result.strip():
                        # Send transcription to main pipeline
                        self.stt_text_queue.put(asyncio_result.strip())
                        
                        # Update timestamp
                        self.last_text_change_timestamp.value = time_module.time()
                        
                        logger.info(f"📝 [WebRTC Audio] Transcribed: '{asyncio_result.strip()}'")
        
        except Exception as e:
            logger.error(f"❌ [WebRTC Audio] Error in frame processing: {e}")
    
    def _is_speech(self, audio_chunk):
        """Simple voice activity detection"""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        max_amplitude = np.max(np.abs(audio_chunk))
        
        # Basic threshold-based detection
        is_speech = max_amplitude > self.silence_threshold
        
        if self.frame_count % 50 == 0:  # Log occasionally
            logger.debug(
                f"🔍 [WebRTC Audio] Speech detection: "
                f"RMS={rms:.4f}, Max={max_amplitude:.4f}, "
                f"Speech={is_speech}"
            )
        
        return is_speech
    
    def _transcribe_chunk(self, audio_chunk, stt_processor):
        """Transcribe audio chunk using STT processor"""
        try:
            # Convert numpy array to WAV bytes (similar to STTAudioProcess)
            import io
            import soundfile as sf
            
            # Create WAV bytes from numpy array
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_chunk, self.sample_rate, format='WAV', subtype='PCM_16')
            wav_bytes = wav_buffer.getvalue()
            
            # Run STT transcription synchronously
            # Note: We need to run the async transcribe method in sync context
            import asyncio
            
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, create a task
                    task = asyncio.create_task(stt_processor.transcribe(wav_bytes))
                    # This won't work in a running loop, need different approach
                    logger.warning("🔄 [WebRTC Audio] Running in async context - skipping transcription")
                    return ""
                else:
                    # Run in existing loop
                    result = loop.run_until_complete(stt_processor.transcribe(wav_bytes))
                    return result
            except RuntimeError:
                # No event loop, create new one
                result = asyncio.run(stt_processor.transcribe(wav_bytes))
                return result
                
        except Exception as e:
            logger.error(f"❌ [WebRTC Audio] Transcription error: {e}")
            return ""


# Utility function to create WebRTC audio queue
def create_webrtc_audio_queue(maxsize=1000):
    """Create a multiprocessing queue for WebRTC audio"""
    return multiprocessing.Queue(maxsize=maxsize)


# Test function for WebRTC audio process
def test_webrtc_audio_process():
    """Test the WebRTC audio process with dummy data"""
    logger.info("🧪 [WebRTC Audio] Testing WebRTC audio process...")
    
    # Create queues
    stt_text_queue = multiprocessing.Queue()
    webrtc_audio_queue = multiprocessing.Queue()
    last_text_change_timestamp = multiprocessing.Value('d', 0.0)
    
    # Create process
    process = WebRTCAudioProcess(
        stt_text_queue=stt_text_queue,
        webrtc_audio_queue=webrtc_audio_queue,
        last_text_change_timestamp=last_text_change_timestamp
    )
    
    try:
        # Start process
        process.start()
        
        # Send test audio data
        sample_rate = 24000
        duration = 2.0  # 2 seconds
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Send audio in chunks
        chunk_size = 1024
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            webrtc_audio_queue.put(chunk)
            time_module.sleep(0.05)  # Simulate real-time
        
        logger.info("🧪 [WebRTC Audio] Test audio sent")
        
        # Wait a bit and check for results
        time_module.sleep(3)
        
        # Check for transcription results
        try:
            while not stt_text_queue.empty():
                result = stt_text_queue.get_nowait()
                logger.info(f"🧪 [WebRTC Audio] Test result: '{result}'")
        except:
            pass
        
    finally:
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        
        logger.info("🧪 [WebRTC Audio] Test complete")


if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    test_webrtc_audio_process()
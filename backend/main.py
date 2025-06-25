"""
AI Order Taker Backend
FastAPI + WebSocket server for real-time audio streaming and order processing
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data models
class OrderItem(BaseModel):
    id: str
    name: str
    price: float
    quantity: int = 1

class Order(BaseModel):
    items: List[OrderItem]
    total: float
    timestamp: datetime

# Global state (in production, use proper state management)
current_order: List[OrderItem] = []
active_connections: List[WebSocket] = []
order_connections: List[WebSocket] = []
audio_accumulator: Dict[str, bytes] = {}  # Store accumulated audio per connection

# Create directories
Path("logs").mkdir(exist_ok=True)
Path("menus").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)
Path("audio_debug").mkdir(exist_ok=True)

app = FastAPI(title="AI Order Taker Backend")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for production)
if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

class LatencyLogger:
    """Simple latency logging utility"""
    
    def __init__(self):
        self.log_file = Path("logs") / f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
    def log_event(self, event_type: str, data: Dict[str, Any], latency_ms: float = None):
        """Log an event with optional latency"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "latency_ms": latency_ms,
            "data": data
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        logger.info(f"{event_type}: {latency_ms}ms" if latency_ms else f"{event_type}")

latency_logger = LatencyLogger()

def convert_webm_to_wav(webm_bytes: bytes) -> bytes:
    """Convert WebM audio bytes to WAV format for STT processing"""
    try:
        # Add ffmpeg flags to handle fragmented/incomplete WebM
        result = subprocess.run([
            'ffmpeg', 
            '-hide_banner', '-loglevel', 'error',  # Reduce noise
            '-i', 'pipe:0',           # Read from stdin
            '-f', 'wav',              # Output WAV format
            '-ac', '1',               # Mono audio
            '-ar', '16000',           # 16kHz sample rate
            '-acodec', 'pcm_s16le',   # 16-bit PCM
            '-fflags', '+genpts',     # Generate timestamps for incomplete streams
            '-avoid_negative_ts', 'make_zero',  # Handle timing issues
            'pipe:1'                  # Write to stdout
        ], 
        input=webm_bytes, 
        capture_output=True,
        check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Log more details about the failure
        stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr"
        logger.error(f"FFmpeg conversion failed (exit {e.returncode}): {stderr_output}")
        return b""

# STT components
import tempfile
from abc import ABC, abstractmethod
from audio_chunking import ChunkProcessor

# ================== STT MODEL SELECTION ==================
# Change this variable to switch between STT models
STT_MODEL = "parakeet"  # Options: "whisper", "parakeet"
# ==========================================================

class BaseSTTProcessor(ABC):
    """Abstract base class for STT processors"""
    
    @abstractmethod
    async def transcribe(self, wav_bytes: bytes) -> str:
        pass

class WhisperSTTProcessor(BaseSTTProcessor):
    """Real-time STT using Faster-Whisper"""
    
    def __init__(self):
        from faster_whisper import WhisperModel
        import os
        
        # Suppress various warning outputs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        logger.info("Loading Faster-Whisper model (tiny.en, GPU optimized, int8)")
        self.model = WhisperModel(
            "tiny.en", 
            device="cuda", 
            compute_type="int8",
            device_index=0,
            cpu_threads=0  # Force GPU-only processing
        )
        logger.info("✅ Faster-Whisper GPU model loaded successfully")
        self.device = "GPU"
        
        # Warm up the model
        logger.info("🔥 Warming up Whisper GPU model...")
        self._warmup_model()
    
    def _warmup_model(self):
        """Warm up the model with actual audio file to avoid cold start"""
        warmup_file = "test/warm_up.wav"
        
        start_time = time.time()
        segments, _ = self.model.transcribe(
            warmup_file,
            beam_size=1,
            best_of=1,
            temperature=0,
            condition_on_previous_text=False,
            word_timestamps=False,
            language="en",
            task="transcribe",
            vad_filter=False,
            vad_parameters=None
        )
        # Force evaluation of all segments
        list(segments)
        warmup_ms = (time.time() - start_time) * 1000
        
        logger.info(f"🚀 GPU warmup completed with {warmup_file} in {warmup_ms:.0f}ms")
		
    
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio bytes to text"""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav_bytes)
            tmp_file.flush()
            
            # Time the complete inference process
            inference_start = time.time()
            segments, info = self.model.transcribe(
                tmp_file.name,
                beam_size=1,           # Fastest decoding
                best_of=1,             # No multiple candidates
                temperature=0,         # Deterministic output
                condition_on_previous_text=True,  # No context carryover
                word_timestamps=False, # Skip word-level timestamps
                language="en",
                task="transcribe",
                vad_filter=False,      # Skip voice activity detection
                vad_parameters=None
            )
            
            # Process segments (where the real work happens)
            text = " ".join([segment.text.strip() for segment in segments])
            inference_ms = (time.time() - inference_start) * 1000
            
        # Cleanup
        import os
        os.unlink(tmp_file.name)
        
        # Calculate performance metrics
        duration_seconds = len(wav_bytes) / (16000 * 2)  # Approximate duration
        realtime_factor = duration_seconds * 1000 / inference_ms
        
        print(f"🎤 STT: {inference_ms:.0f}ms ({realtime_factor:.1f}x realtime) → '{text}'")
        
        latency_logger.log_event("STT_TRANSCRIBE", {
            "text": text, 
            "inference_ms": inference_ms,
            "realtime_factor": realtime_factor
        }, inference_ms)
        
        return text.strip()

class ParakeetSTTProcessor(BaseSTTProcessor):
    """Real-time STT using Parakeet (NeMo ASR) with proper streaming support"""
    
    def __init__(self):
        import os
        import torch
        from typing import Any
        
        # Remove environment variable overhead (not in nemo_streaming_test.py)
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        # os.environ['HYDRA_FULL_ERROR'] = '1'
        
        # Remove PyTorch optimizations that may be causing overhead
        # logger.info("🚀 Enabling PyTorch optimizations...")
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = False
        # torch._C._jit_set_profiling_executor(False)
        # torch._C._jit_set_profiling_mode(False)
        
        logger.info("Loading Parakeet model (NeMo ASR, PyTorch optimized)")
        
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            
            # Load model using generic ASRModel (like nemo_streaming_test.py)
            logger.info("🔄 Loading FastConformer Transducer model...")
            model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
            self.model: Any = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

            # Configure decoding exactly like nemo_streaming_test.py
            from omegaconf import open_dict
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"  # Use greedy instead of greedy_batch
                decoding_cfg.preserve_alignments = False
                if hasattr(self.model, 'joint'):  # if an RNNT model
                    decoding_cfg.fused_batch_size = -1
                    if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                        decoding_cfg.greedy.max_symbols = 10
                if hasattr(self.model, "cur_decoder"):
                    # hybrid model, explicitly pass decoder type, otherwise it will be set to "rnnt"
                    self.model.change_decoding_strategy(decoding_cfg, decoder_type=self.model.cur_decoder)
                else:
                    self.model.change_decoding_strategy(decoding_cfg)
            
            # Store the buffer class for later use
            self.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer

            self.model.eval()
            self.model = self.model.cuda()
            
            # Remove GPU memory monitoring (adds overhead)
            # if torch.cuda.is_available():
            #     memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            #     memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            #     logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            logger.info("✅ FastConformer Streaming model loaded successfully (GPU optimized)")
            self.device = "GPU"
            
            # Warm up the model
            logger.info("🔥 Warming up FastConformer GPU model...")
            self._warmup_model()
            
        except ImportError as e:
            logger.error("❌ NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load Parakeet model: {e}")
            raise e
    
    def _warmup_model(self):
        """Warm up the model with actual audio file to avoid cold start"""
        import torch
        
        warmup_file = "test/warm_up.wav"
        
        try:
            start_time = time.time()
            # GPU cache management (empty_cache removed - was causing CUDA errors)
            
            # NeMo transcription (remove torch.no_grad and synchronize overhead)
            text = self.model.transcribe([warmup_file], verbose=False)
            
            # Remove torch.cuda.synchronize() - adds overhead
            # torch.cuda.synchronize()
            warmup_ms = (time.time() - start_time) * 1000
            
            logger.info(f"🚀 FastConformer GPU warmup completed with {warmup_file} in {warmup_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"Warmup failed: {e}")
            # Don't crash on warmup failure, just log it
            logger.warning("Continuing without warmup")
    
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio using NeMo CacheAwareStreamingAudioBuffer like nemo_streaming_test.py"""
        import torch
        from pathlib import Path
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        inference_start = time.time()
        
        # Use the test audio file directly (like nemo_streaming_test.py)
        audio_file = "test/test_audio.wav"
        
        if not Path(audio_file).exists():
            print(f"❌ Audio file not found: {audio_file}")
            return ""
        
        print(f"📁 Loading audio file: {audio_file}")
        
        # Initialize streaming buffer (exactly like nemo_streaming_test.py)
        streaming_buffer = self.CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        
        # Add audio file to buffer (exactly like nemo_streaming_test.py)
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            audio_file, stream_id=-1
        )
        
        # Set up autocast exactly like nemo_streaming_test.py  
        autocast = torch.amp.autocast(self.model.device.type, enabled=True)
        
        # Helper functions (exactly from nemo_streaming_test.py)
        def extract_transcriptions(hyps):
            """
            The transcribed_texts returned by CTC and RNNT models are different.
            This method would extract and return the text section of the hypothesis.
            """
            if isinstance(hyps[0], Hypothesis):
                transcriptions = []
                for hyp in hyps:
                    transcriptions.append(hyp.text)
            else:
                transcriptions = hyps
            return transcriptions

        def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
            # for the first step there is no need to drop any tokens after the downsampling as no caching is being used
            if step_num == 0 and not pad_and_drop_preencoded:
                return 0
            else:
                return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

        def perform_streaming(asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False):
            batch_size = len(streaming_buffer.streams_length)
            if compare_vs_offline:
                # would pass the whole audio at once through the model like offline mode in order to compare the results with the stremaing mode
                # the output of the model in the offline and streaming mode should be exactly the same
                with torch.inference_mode():
                    with autocast:
                        processed_signal, processed_signal_length = streaming_buffer.get_all_audios()
                        with torch.no_grad():
                            (
                                pred_out_offline,
                                transcribed_texts,
                                cache_last_channel_next,
                                cache_last_time_next,
                                cache_last_channel_len,
                                best_hyp,
                            ) = asr_model.conformer_stream_step(
                                processed_signal=processed_signal,
                                processed_signal_length=processed_signal_length,
                                return_transcription=True,
                            )
                final_offline_tran = extract_transcriptions(transcribed_texts)
                print(f" Final offline transcriptions:   {final_offline_tran}")
            else:
                final_offline_tran = None

            cache_last_channel, cache_last_time, cache_last_channel_len = asr_model.encoder.get_initial_cache_state(
                batch_size=batch_size
            )

            previous_hypotheses = None
            streaming_buffer_iter = iter(streaming_buffer)
            pred_out_stream = None
            
            print("🔄 Starting streaming inference...")
            start_time = time.time()
            
            for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
                step_start = time.time()
                with torch.inference_mode():
                    with autocast:
                        # keep_all_outputs needs to be True for the last step of streaming when model is trained with att_context_style=regular
                        # otherwise the last outputs would get dropped

                        with torch.no_grad():
                            (
                                pred_out_stream,
                                transcribed_texts,
                                cache_last_channel,
                                cache_last_time,
                                cache_last_channel_len,
                                previous_hypotheses,
                            ) = asr_model.conformer_stream_step(
                                processed_signal=chunk_audio,
                                processed_signal_length=chunk_lengths,
                                cache_last_channel=cache_last_channel,
                                cache_last_time=cache_last_time,
                                cache_last_channel_len=cache_last_channel_len,
                                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                                previous_hypotheses=previous_hypotheses,
                                previous_pred_out=pred_out_stream,
                                drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                                    asr_model, step_num, pad_and_drop_preencoded
                                ),
                                return_transcription=True,
                            )

                step_ms = (time.time() - step_start) * 1000
                if debug_mode:
                    current_transcripts = extract_transcriptions(transcribed_texts)
                    print(f"⏱️  Step {step_num}: {step_ms:.0f}ms -> {current_transcripts}")

            total_time = time.time() - start_time
            final_streaming_tran = extract_transcriptions(transcribed_texts)
            print(f"✅ Final streaming transcriptions: {final_streaming_tran}")
            print(f"🎤 Total streaming time: {total_time*1000:.0f}ms")

            if compare_vs_offline:
                # calculates and report the differences between the predictions of the model in offline mode vs streaming mode
                # Normally they should be exactly the same predictions for streaming models
                pred_out_stream_cat = torch.cat(pred_out_stream)
                pred_out_offline_cat = torch.cat(pred_out_offline)
                if pred_out_stream_cat.size() == pred_out_offline_cat.size():
                    diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
                    print(f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode.")
                else:
                    print(f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()}).")

            return final_streaming_tran, final_offline_tran
        
        # Call perform_streaming exactly like nemo_streaming_test.py
        final_streaming_tran, final_offline_tran = perform_streaming(
            asr_model=self.model,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=False,  # Set to True to compare with offline mode
            debug_mode=True,  # Show step-by-step results
            pad_and_drop_preencoded=False,
        )
        
        # Get final transcription
        final_text = final_streaming_tran[0] if final_streaming_tran else ""
        
        total_inference_ms = (time.time() - inference_start) * 1000
        
        print(f"✅ Final streaming transcription: {final_streaming_tran}")
        print(f"🎤 Total streaming time: {total_inference_ms:.0f}ms")
        
        # Calculate performance metrics 
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_file)
            audio_duration = len(audio_data) / sample_rate
            realtime_factor = total_inference_ms / 1000 / audio_duration
            print(f"🚀 Real-time factor: {realtime_factor:.2f}x")
        except Exception as e:
            print(f"⚠️  Could not calculate RTF: {e}")
            realtime_factor = 0
        
        latency_logger.log_event("STT_TRANSCRIBE", {
            "model": "fastconformer-streaming-nemo-official",
            "text": final_text, 
            "inference_ms": total_inference_ms,
            "realtime_factor": realtime_factor
        }, total_inference_ms)
        
        return final_text.strip()

# ================== STT FACTORY ==================
def create_stt_processor() -> BaseSTTProcessor:
    """Factory function to create the selected STT processor"""
    if STT_MODEL == "whisper":
        return WhisperSTTProcessor()
    elif STT_MODEL == "parakeet":
        return ParakeetSTTProcessor()
    else:
        raise ValueError(f"Unknown STT model: {STT_MODEL}. Options: 'whisper', 'parakeet'")

class LLMReasoner:
    """Stub for Phi-3 Mini reasoning LLM"""
    
    def __init__(self):
        self.menu_context = self.load_menu_context()
    
    def load_menu_context(self) -> str:
        """Load menu context from uploaded files"""
        menu_files = list(Path("menus").glob("*.txt"))
        if not menu_files:
            return "Default menu: Cheeseburger ($8.99), Fries ($3.99), Drink ($2.99)"
        
        context = ""
        for file in menu_files:
            context += file.read_text() + "\n"
        return context
    
    async def process_order(self, text: str) -> List[OrderItem]:
        """Process user text into order items"""
        start_time = time.time()
        
        # Simulate LLM processing
        await asyncio.sleep(0.2)
        
        # Mock order processing
        items = [
            OrderItem(id="1", name="Cheeseburger", price=8.99),
            OrderItem(id="2", name="Fries", price=3.99)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("LLM_REASONING", {
            "input_text": text,
            "output_items": [item.dict() for item in items]
        }, latency_ms)
        
        return items
    
    async def generate_response(self, order_items: List[OrderItem]) -> str:
        """Generate response text for TTS"""
        start_time = time.time()
        
        await asyncio.sleep(0.1)
        
        response = f"I've added {len(order_items)} items to your order. Anything else?"
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("LLM_RESPONSE", {
            "response_text": response
        }, latency_ms)
        
        return response

class TTSProcessor:
    """Stub for Chatterbox TTS"""
    
    async def synthesize(self, text: str) -> bytes:
        """Stub TTS synthesis"""
        start_time = time.time()
        
        await asyncio.sleep(0.15)
        
        # Mock audio data
        audio_data = b"mock_audio_data"
        
        latency_ms = (time.time() - start_time) * 1000
        latency_logger.log_event("TTS_SYNTHESIS", {
            "text": text,
            "audio_length": len(audio_data)
        }, latency_ms)
        
        return audio_data

# Initialize processors
stt_processor = create_stt_processor()
llm_reasoner = LLMReasoner()
tts_processor = TTSProcessor()

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket endpoint for audio streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    connection_id = str(id(websocket))
    audio_accumulator[connection_id] = b""
    
    logger.info("Audio WebSocket connected")
    
    try:
        while True:
            # === ORIGINAL AUDIO PROCESSING (COMMENTED OUT FOR TESTING) ===
            # # Receive audio chunk
            # webm_chunk = await websocket.receive_bytes()
            # 
            # # Accumulate chunks
            # audio_accumulator[connection_id] += webm_chunk
            # accumulated_audio = audio_accumulator[connection_id]
            # 
            # print(f"Accumulating... Total WebM bytes: {len(accumulated_audio)}")
            # 
            # # Try to convert accumulated audio every 3 chunks (to reduce processing load)
            # chunk_count = len(accumulated_audio) // 16422  # Approximate chunks received
            # 
            # if chunk_count > 0 and chunk_count % 3 == 0:  # Every 3 chunks
            #     wav_chunk = convert_webm_to_wav(accumulated_audio)
            #     
            #     if wav_chunk:
            #         # Save WAV chunk for debugging
            #         timestamp = int(time.time() * 1000)
            #         filename = f"audio_accumulated_{timestamp}.wav"
            #         with open(f"audio_debug/{filename}", "wb") as f:
            #             f.write(wav_chunk)
            #         
            #         print(f"CONVERSION SUCCESS: WebM {len(accumulated_audio)} bytes → WAV {len(wav_chunk)} bytes")
            #         print(f"WAV header: {wav_chunk[:12]}")
            #         
            #         # Process audio through pipeline
            #         await process_audio_pipeline(wav_chunk, websocket)
            #     else:
            #         print(f"Conversion failed for {len(accumulated_audio)} bytes")
            # 
            # # Log audio received
            # latency_logger.log_event("AUDIO_RECEIVED", {
            #     "chunk_size": len(webm_chunk),
            #     "accumulated_size": len(accumulated_audio),
            #     "estimated_chunks": chunk_count
            # })
            
            # === TEST MODE: USE STATIC WAV FILE ===
            # Still receive bytes to maintain WebSocket protocol
            await websocket.receive_bytes()
            
            # Load test WAV file instead of processing real audio
            test_wav_path = "test/test_audio.wav"
            try:
                with open(test_wav_path, "rb") as f:
                    wav_bytes = f.read()
                
                print(f"📁 Loading test file: {test_wav_path} ({len(wav_bytes)} bytes)")
                
                # Process test audio through STT pipeline
                await process_audio_pipeline(wav_bytes, websocket)
                
                # Log test audio processing
                latency_logger.log_event("AUDIO_RECEIVED", {
                    "test_file": test_wav_path,
                    "wav_size": len(wav_bytes)
                })
                
            except FileNotFoundError:
                print(f"❌ Test file not found: {test_wav_path}")
                print("💡 Create a test WAV file at test/test_audio.wav")
                await asyncio.sleep(1)  # Prevent spam
            
    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
        if websocket in active_connections:
            active_connections.remove(websocket)
        # Clean up accumulator
        if connection_id in audio_accumulator:
            del audio_accumulator[connection_id]
    except Exception as e:
        logger.error(f"Error in audio WebSocket: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
        if connection_id in audio_accumulator:
            del audio_accumulator[connection_id]

@app.websocket("/ws/order")
async def websocket_order(websocket: WebSocket):
    """WebSocket endpoint for order updates"""
    await websocket.accept()
    order_connections.append(websocket)
    
    logger.info("Order WebSocket connected")
    
    try:
        # Send current order on connect
        order_data = {
            "items": [item.dict() for item in current_order],
            "total": sum(item.price * item.quantity for item in current_order)
        }
        await websocket.send_text(json.dumps(order_data))
        
        # Keep connection alive
        while True:
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info("Order WebSocket disconnected")
        if websocket in order_connections:
            order_connections.remove(websocket)

async def process_audio_pipeline(audio_chunk: bytes, websocket: WebSocket):
    """Process audio through the full pipeline"""
    pipeline_start = time.time()
    
    try:
        # Step 1: STT
        transcribed_text = await stt_processor.transcribe(audio_chunk)
        
        # Step 2: LLM reasoning
        new_items = await llm_reasoner.process_order(transcribed_text)
        
        # Step 3: Update order
        global current_order
        current_order.extend(new_items)
        
        # Step 4: Generate response
        response_text = await llm_reasoner.generate_response(new_items)
        
        # Step 5: TTS
        audio_response = await tts_processor.synthesize(response_text)
        
        # Step 6: Send updates
        await broadcast_order_update()
        
        # Send transcription back
        try:
            await websocket.send_text(json.dumps({
                "type": "transcription",
                "text": transcribed_text
            }))
            
            # Send audio response (stub for now)
            await websocket.send_text(json.dumps({
                "type": "audio_response",
                "text": response_text
            }))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            raise  # Re-raise to trigger disconnection cleanup
        
        # Log total pipeline latency
        total_latency = (time.time() - pipeline_start) * 1000
        latency_logger.log_event("PIPELINE_TOTAL", {
            "transcription": transcribed_text,
            "response": response_text,
            "items_added": len(new_items)
        }, total_latency)
        
    except Exception as e:
        logger.error(f"Error in audio pipeline: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

async def broadcast_order_update():
    """Broadcast order updates to all connected clients"""
    order_data = {
        "items": [item.dict() for item in current_order],
        "total": sum(item.price * item.quantity for item in current_order)
    }
    
    # Send to all connected order WebSocket clients
    disconnected_connections = []
    for connection in order_connections:
        try:
            await connection.send_text(json.dumps(order_data))
        except Exception as e:
            logger.error(f"Error broadcasting to order connection: {e}")
            disconnected_connections.append(connection)
    
    # Remove disconnected connections
    for connection in disconnected_connections:
        if connection in order_connections:
            order_connections.remove(connection)

@app.post("/api/upload-menu")
async def upload_menu(file: UploadFile = File(...)):
    """Upload menu file (text or image)"""
    try:
        # Save uploaded file
        file_path = Path("uploads") / file.filename
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Process based on file type
        if file.content_type.startswith("text/"):
            # Text file - save directly to menus
            menu_path = Path("menus") / f"{file.filename}.txt"
            with open(menu_path, "wb") as f:
                f.write(content)
            
            logger.info(f"Saved text menu: {menu_path}")
            
        elif file.content_type.startswith("image/"):
            # Image file - stub OCR processing
            logger.info(f"Image uploaded: {file_path} (OCR processing stubbed)")
            
            # Stub: In real implementation, use OCR to extract text
            extracted_text = "Menu extracted from image (stub)"
            menu_path = Path("menus") / f"{file.filename}.txt"
            with open(menu_path, "w") as f:
                f.write(extracted_text)
        
        # Reload menu context
        llm_reasoner.menu_context = llm_reasoner.load_menu_context()
        
        return {"message": "Menu uploaded successfully", "filename": file.filename}
        
    except Exception as e:
        logger.error(f"Error uploading menu: {e}")
        return {"error": str(e)}

@app.get("/api/order")
async def get_current_order():
    """Get current order"""
    return {
        "items": [item.dict() for item in current_order],
        "total": sum(item.price * item.quantity for item in current_order)
    }

@app.post("/api/order/clear")
async def clear_order():
    """Clear current order"""
    global current_order
    current_order = []
    await broadcast_order_update()
    return {"message": "Order cleared"}

@app.get("/")
async def root():
    """Serve the React app in production, API info in development"""
    # Check if we have static files (production mode)
    static_dir = Path("static")
    if static_dir.exists() and (static_dir / "index.html").exists():
        from fastapi.responses import FileResponse
        return FileResponse("static/index.html")
    else:
        # Development mode - just return API info
        return {"message": "AI Order Taker Backend - Development Mode"}

# Catch-all route for React Router (must be last)
@app.get("/{path:path}")
async def serve_react_app(path: str):
    """Serve React app for any non-API route"""
    static_dir = Path("static")
    if static_dir.exists() and (static_dir / "index.html").exists():
        from fastapi.responses import FileResponse
        # Try to serve the specific file first
        file_path = static_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        # Otherwise serve index.html (React Router will handle routing)
        return FileResponse("static/index.html")
    else:
        # Development mode - return 404
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable auto-reload to prevent multiple model loads
        log_level="info"
    )
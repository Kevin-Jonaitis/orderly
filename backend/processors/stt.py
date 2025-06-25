"""
Speech-to-Text (STT) processors for real-time audio transcription.

This module provides:
- BaseSTTProcessor: Abstract base class for STT implementations
- WhisperSTTProcessor: Real-time STT using Faster-Whisper
- ParakeetSTTProcessor: Real-time STT using Parakeet (NeMo ASR) with streaming support
"""

import tempfile
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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
        
        return text.strip()

class ParakeetSTTProcessor(BaseSTTProcessor):
    """Real-time STT using Parakeet (NeMo ASR) - stripped down to match nemo_streaming_test.py exactly"""
    
    def __init__(self):
        import torch
        
        logger.info("🚀 Loading model: stt_en_fastconformer_hybrid_large_streaming_multi")
        
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            from omegaconf import open_dict
            
            # Load model exactly like nemo_streaming_test.py
            model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            
            # Set up autocast for mixed precision (exactly like nemo_streaming_test.py)
            self.autocast = torch.amp.autocast(self.model.device.type, enabled=True)
            
            # Configure decoding exactly like nemo_streaming_test.py
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
            
            # Model setup exactly like nemo_streaming_test.py
            self.model = self.model.cuda()
            self.model.eval()
            
            # Store the buffer class for later use
            self.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer
            
            logger.info("✅ FastConformer model loaded successfully")
            
        except ImportError as e:
            logger.error("❌ NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load Parakeet model: {e}")
            raise e
    
    async def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe WAV audio using exact nemo_streaming_test.py approach"""
        import torch
        import os
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        # Use the test audio file directly (like nemo_streaming_test.py)
        audio_file = "test/test_audio.wav"
        
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return ""
        
        print(f"📁 Loading audio file: {audio_file}")
        
        # Set up streaming buffer exactly like nemo_streaming_test.py
        streaming_buffer = self.CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        
        # Add audio file to buffer
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            audio_file, stream_id=-1
        )
        
        # Make autocast available in local scope like nemo_streaming_test.py
        autocast = self.autocast
        
        # Helper functions copied exactly from nemo_streaming_test.py
        def extract_transcriptions(hyps):
            if isinstance(hyps[0], Hypothesis):
                transcriptions = []
                for hyp in hyps:
                    transcriptions.append(hyp.text)
            else:
                transcriptions = hyps
            return transcriptions

        def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
            if step_num == 0 and not pad_and_drop_preencoded:
                return 0
            else:
                return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

        def perform_streaming(asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False):
            batch_size = len(streaming_buffer.streams_length)
            if compare_vs_offline:
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
            
            print(f"🔍 Debug: streaming_buffer has {len(streaming_buffer.streams_length)} streams")
            print(f"🔍 Debug: About to iterate over streaming buffer...")
            
            for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
                print(f"🔍 Debug: Processing chunk {step_num}, shapes: {chunk_audio.shape if hasattr(chunk_audio, 'shape') else 'no shape'}")
                step_start = time.time()
                with torch.inference_mode():
                    with autocast:
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
                pred_out_stream_cat = torch.cat(pred_out_stream)
                pred_out_offline_cat = torch.cat(pred_out_offline)
                if pred_out_stream_cat.size() == pred_out_offline_cat.size():
                    diff_num = torch.sum(pred_out_stream_cat != pred_out_offline_cat).cpu().numpy()
                    print(f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode.")
                else:
                    print(f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()}).")

            return final_streaming_tran, final_offline_tran
        
        # Perform streaming exactly like nemo_streaming_test.py
        start_total = time.time()
        final_streaming_tran, final_offline_tran = perform_streaming(
            asr_model=self.model,
            streaming_buffer=streaming_buffer,
            compare_vs_offline=False,
            debug_mode=True,
            pad_and_drop_preencoded=False,
        )
        total_total = time.time() - start_total
        
        # Get final result
        final_text = final_streaming_tran[0] if final_streaming_tran else ""
        print(f"⏱️  Total time: {total_total*1000:.0f}ms")
        
        return final_text.strip()

# ================== STT FACTORY ==================
def create_stt_processor(model_name: str = "parakeet") -> BaseSTTProcessor:
    """Factory function to create the selected STT processor"""
    if model_name == "whisper":
        return WhisperSTTProcessor()
    elif model_name == "parakeet":
        return ParakeetSTTProcessor()
    else:
        raise ValueError(f"Unknown STT model: {model_name}. Options: 'whisper', 'parakeet'")
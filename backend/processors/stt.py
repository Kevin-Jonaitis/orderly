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

class RealTimeSTTProcessor(BaseSTTProcessor):
    """Real-time STT using NeMo FastConformer with streaming support"""
    
    # Latency options for att_context_size:
    # [70, 1] = 80ms latency  
    # [70, 16] = 480ms latency
    # [70, 33] = 1040ms latency
    ATT_CONTEXT_SIZE = [70, 1]  # 80ms latency
    
    def __init__(self):
        import torch
        import numpy as np
        
        logger.info("🚀 Loading model: stt_en_fastconformer_hybrid_large_streaming_multi")
        
        try:
            import nemo.collections.asr as nemo_asr
            from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            from omegaconf import open_dict
            
            # Load model
            model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
            
            # Set up autocast for mixed precision
            self.autocast = torch.amp.autocast(self.model.device.type, enabled=True)
            
            # Configure decoding
            decoding_cfg = self.model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.strategy = "greedy"
                decoding_cfg.preserve_alignments = False
                if hasattr(self.model, 'joint'):
                    decoding_cfg.fused_batch_size = -1
                    if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                        decoding_cfg.greedy.max_symbols = 10
                if hasattr(self.model, "cur_decoder"):
                    self.model.change_decoding_strategy(decoding_cfg, decoder_type=self.model.cur_decoder)
                else:
                    self.model.change_decoding_strategy(decoding_cfg)
            
            # Model setup - GPU only
            self.model = self.model.to('cuda')
            self.model.eval()
            
            # Store buffer class
            self.CacheAwareStreamingAudioBuffer = CacheAwareStreamingAudioBuffer
            
            # Set attention context size for low latency
            logger.info(f"🔧 Setting att_context_size to {self.ATT_CONTEXT_SIZE} (80ms latency)")
            self.model.encoder.set_default_att_context_size(self.ATT_CONTEXT_SIZE)
            
            # Initialize streaming state
            self.reset_state()
            
            # Audio buffering (160ms minimum for stable mel features)
            self.audio_buffer = []
            self.min_chunks = 2  # 160ms minimum (2 * 80ms chunks)
            
            logger.info("✅ FastConformer model loaded successfully")
            
        except ImportError as e:
            logger.error("❌ NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e
    
    def reset_state(self):
        """Reset streaming state between sessions"""
        import torch
        batch_size = 1
        self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len = \
            self.model.encoder.get_initial_cache_state(batch_size=batch_size)
        self.previous_hypotheses = None
        self.pred_out_stream = None
        self.step_num = 0
        
    async def transcribe(self, wav_bytes: bytes) -> str:
        """File-based transcription for compatibility with BaseSTTProcessor"""
        import torch
        import os
        import tempfile
        from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
        
        # Save wav_bytes to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(wav_bytes)
            tmp_file.flush()
            audio_file = tmp_file.name
        
        if not os.path.exists(audio_file):
            logger.error(f"❌ Audio file not found: {audio_file}")
            return ""
        
        # Set up streaming buffer
        streaming_buffer = self.CacheAwareStreamingAudioBuffer(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        
        # Add audio file to buffer
        processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
            audio_file, stream_id=-1
        )
        
        # Helper functions
        def extract_transcriptions(hyps):
            if isinstance(hyps[0], Hypothesis):
                return [hyp.text for hyp in hyps]
            else:
                return hyps

        def calc_drop_extra_pre_encoded(asr_model, step_num, pad_and_drop_preencoded):
            if step_num == 0 and not pad_and_drop_preencoded:
                return 0
            else:
                return asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

        # Perform streaming inference
        cache_last_channel, cache_last_time, cache_last_channel_len = \
            self.model.encoder.get_initial_cache_state(batch_size=1)
        previous_hypotheses = None
        streaming_buffer_iter = iter(streaming_buffer)
        pred_out_stream = None
        
        start_time = time.time()
        
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
            with torch.inference_mode():
                with self.autocast:
                    with torch.no_grad():
                        (
                            pred_out_stream,
                            transcribed_texts,
                            cache_last_channel,
                            cache_last_time,
                            cache_last_channel_len,
                            previous_hypotheses,
                        ) = self.model.conformer_stream_step(
                            processed_signal=chunk_audio,
                            processed_signal_length=chunk_lengths,
                            cache_last_channel=cache_last_channel,
                            cache_last_time=cache_last_time,
                            cache_last_channel_len=cache_last_channel_len,
                            keep_all_outputs=streaming_buffer.is_buffer_empty(),
                            previous_hypotheses=previous_hypotheses,
                            previous_pred_out=pred_out_stream,
                            drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                                self.model, step_num, False
                            ),
                            return_transcription=True,
                        )

        total_time = (time.time() - start_time) * 1000
        final_streaming_tran = extract_transcriptions(transcribed_texts)
        
        # Get final result
        if final_streaming_tran and len(final_streaming_tran) > 0:
            final_text = final_streaming_tran[0] if isinstance(final_streaming_tran[0], str) else str(final_streaming_tran[0])
        else:
            final_text = ""
        
        logger.info(f"🎤AUDIO STREAMER STT: {total_time:.0f}ms → '{final_text.strip()}'")
        
        # Cleanup
        try:
            os.unlink(audio_file)
        except:
            pass
        
        return final_text.strip()
    
    async def process_audio_chunk(self, audio_chunk):
        """Process 80ms audio chunks for real-time streaming"""
        import torch
        import numpy as np
        from datetime import datetime
        
        try:
            # We don't need to downsample here, because we're already doing it in the webrtc.py file
            # # Downsample to 16kHz for NeMo
            # audio_16k = self.downsample_audio(audio_chunk)
            
            # Add to buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Process when we have 2 chunks (160ms) for stable mel features
            chunk_samples = len(audio_chunk)
            min_samples = chunk_samples * self.min_chunks  # 160ms worth
            
            if len(self.audio_buffer) < min_samples:
                # Need more chunks for stable processing
                return "", 0.0
            
            # Take exactly 2 chunks worth of audio
            audio_to_process = np.array(self.audio_buffer[:min_samples])
            # Remove the processed audio, keep remainder
            self.audio_buffer = self.audio_buffer[chunk_samples:]  # Remove 1 chunk, keep overlap
            
            # Apply preprocessing to match NeMo's expected format  
            audio_tensor = torch.tensor(audio_to_process, dtype=torch.float32).unsqueeze(0)  # (1, time)
            audio_length = torch.tensor([len(audio_to_process)], dtype=torch.long)
            
            # Move tensors to the same device as the model
            device = next(self.model.parameters()).device
            audio_tensor = audio_tensor.to(device)
            audio_length = audio_length.to(device)
            
            # Use model's preprocessor to get features
            with torch.no_grad():
                processed_signal, processed_signal_length = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_length
                )
            
            # Check if we got valid features
            if processed_signal.numel() == 0 or processed_signal_length.item() == 0:
                return "", 0.0
            
            # Process with streaming model
            start_time = time.time()
            
            with torch.inference_mode():
                with self.autocast:
                    with torch.no_grad():
                        (
                            self.pred_out_stream,
                            transcribed_texts,
                            self.cache_last_channel,
                            self.cache_last_time,
                            self.cache_last_channel_len,
                            self.previous_hypotheses,
                        ) = self.model.conformer_stream_step(
                            processed_signal=processed_signal,
                            processed_signal_length=processed_signal_length,
                            cache_last_channel=self.cache_last_channel,
                            cache_last_time=self.cache_last_time,
                            cache_last_channel_len=self.cache_last_channel_len,
                            keep_all_outputs=True,
                            previous_hypotheses=self.previous_hypotheses,
                            previous_pred_out=self.pred_out_stream,
                            drop_extra_pre_encoded=0 if self.step_num == 0 else self.model.encoder.streaming_cfg.drop_extra_pre_encoded,
                            return_transcription=True,
                        )
            
            process_time = (time.time() - start_time) * 1000

            # Extract transcription
            if transcribed_texts and len(transcribed_texts) > 0:
                if hasattr(transcribed_texts[0], 'text'):
                    current_text = transcribed_texts[0].text
                else:
                    current_text = str(transcribed_texts[0])
            else:
                current_text = ""
            
            # Log results
            if current_text.strip():
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🎤 AUDIO OUTPUT [{self.step_num:3d}] {process_time:3.0f}ms: '{current_text}'")
            elif self.step_num % 10 == 0:  # Show progress every 10 processed chunks 
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] 🔄 [{self.step_num:3d}] Processing...")
            
            self.step_num += 1
            return current_text.strip(), process_time
            
        except Exception as e:
            print(f"[NeMo ERROR] Exception in process_audio_chunk: {e}")
            import traceback
            traceback.print_exc()
            return "", 0.0

# ================== STT FACTORY ==================
def create_stt_processor(model_name: str = "realtime") -> BaseSTTProcessor:
    """Factory function to create the selected STT processor"""
    if model_name == "realtime":
        return RealTimeSTTProcessor()
    else:
        raise ValueError(f"Unknown STT model: {model_name}. Only 'realtime' is supported.")
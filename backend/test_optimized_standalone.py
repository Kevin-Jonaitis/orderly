#!/usr/bin/env python3
"""
Standalone test of optimized ParakeetSTTProcessor (extracted from main.py)
"""

import time
import torch
from typing import Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedParakeetSTTProcessor:
    """Real-time STT using Parakeet (NeMo ASR) with optimized model loading"""
    
    def __init__(self):
        import os
        import torch
        from typing import Any
        
        logger.info("Loading Parakeet model (NeMo ASR, optimized for streaming)")
        
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

            # Set up model order like nemo_streaming_test.py: cuda() then eval()
            self.model = self.model.cuda()
            self.model.eval()
            
            # Set up global autocast like nemo_streaming_test.py
            self.autocast = torch.amp.autocast(self.model.device.type, enabled=True)
            
            logger.info("✅ FastConformer Streaming model loaded successfully (GPU optimized)")
            self.device = "GPU"
            
            # Streaming cache warmup instead of non-streaming warmup
            logger.info("🔥 Warming up streaming cache states...")
            self._warmup_streaming_cache()
            
        except ImportError as e:
            logger.error("❌ NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
            raise e
        except Exception as e:
            logger.error(f"❌ Failed to load Parakeet model: {e}")
            raise e
    
    def _warmup_streaming_cache(self):
        """Warm up streaming cache states to avoid first-chunk overhead"""
        import torch
        
        try:
            start_time = time.time()
            
            # Initialize streaming cache states like nemo_streaming_test.py
            batch_size = 1  # Single stream
            cache_last_channel, cache_last_time, cache_last_channel_len = self.model.encoder.get_initial_cache_state(
                batch_size=batch_size
            )
            
            # Create a small dummy audio tensor to warm up the inference path
            # This matches the streaming buffer approach used in transcribe()
            dummy_audio = torch.zeros((batch_size, 1600), device=self.model.device)  # 100ms at 16kHz
            dummy_lengths = torch.tensor([1600], device=self.model.device)
            
            with torch.inference_mode():
                with self.autocast:
                    with torch.no_grad():
                        # Run one step of streaming inference to warm up cache
                        (
                            pred_out_stream,
                            transcribed_texts,
                            cache_last_channel,
                            cache_last_time,
                            cache_last_channel_len,
                            previous_hypotheses,
                        ) = self.model.conformer_stream_step(
                            processed_signal=dummy_audio,
                            processed_signal_length=dummy_lengths,
                            cache_last_channel=cache_last_channel,
                            cache_last_time=cache_last_time,
                            cache_last_channel_len=cache_last_channel_len,
                            keep_all_outputs=True,
                            previous_hypotheses=None,
                            previous_pred_out=None,
                            drop_extra_pre_encoded=0,
                            return_transcription=True,
                        )
            
            warmup_ms = (time.time() - start_time) * 1000
            logger.info(f"🚀 Streaming cache warmup completed in {warmup_ms:.0f}ms")
            
        except Exception as e:
            logger.error(f"Streaming warmup failed: {e}")
            logger.warning("Continuing without streaming warmup")
    
    def transcribe(self, wav_bytes: bytes) -> str:
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
        
        # Use the global autocast set up during initialization
        
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
                    with self.autocast:
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
                    with self.autocast:
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
        
        return final_text.strip()

def main():
    print("🧪 Testing optimized ParakeetSTTProcessor...")
    
    # Initialize processor
    print("🚀 Initializing processor...")
    start_init = time.time()
    processor = OptimizedParakeetSTTProcessor()
    init_time = (time.time() - start_init) * 1000
    print(f"⏱️  Initialization took: {init_time:.0f}ms")
    
    # Test transcription
    print("🎯 Testing transcription...")
    start_total = time.time()
    
    # Load test audio file
    with open("test/test_audio.wav", "rb") as f:
        wav_bytes = f.read()
    
    # Transcribe
    result = processor.transcribe(wav_bytes)
    
    total_time = (time.time() - start_total) * 1000
    print(f"🎤 Final result: '{result}'")
    print(f"⏱️  Total time: {total_time:.0f}ms")

if __name__ == "__main__":
    main()
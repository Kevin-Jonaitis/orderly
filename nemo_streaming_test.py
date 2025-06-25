#!/usr/bin/env python3
"""
Test NeMo cache-aware streaming with our model and audio file.
Adapted from NVIDIA NeMo official streaming script.
"""

import contextlib
import json
import os
import time
from argparse import ArgumentParser

import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.utils import logging


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


def perform_streaming(
    asr_model, streaming_buffer, compare_vs_offline=False, debug_mode=False, pad_and_drop_preencoded=False
):
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
        logging.info(f" Final offline transcriptions:   {final_offline_tran}")
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
            logging.info(
                f"Found {diff_num} differences in the outputs of the model in streaming mode vs offline mode."
            )
        else:
            logging.info(
                f"The shape of the outputs of the model in streaming mode ({pred_out_stream_cat.size()}) is different from offline mode ({pred_out_offline_cat.size()})."
            )

    return final_streaming_tran, final_offline_tran


def main():
    print("🧪 Testing NeMo Cache-Aware Streaming...")
    
    # Set up model
    model_name = "stt_en_fastconformer_hybrid_large_streaming_multi"
    audio_file = "backend/test/test_audio.wav"
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🚀 Loading model: {model_name}")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    
    print("📊 Model streaming config:")
    print(asr_model.encoder.streaming_cfg)
    
    # Set up autocast for mixed precision
    global autocast
    autocast = torch.amp.autocast(asr_model.device.type, enabled=True)
    
    # Configure decoding - use greedy instead of greedy_batch to avoid partial hypotheses issues
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"  # Use greedy instead of greedy_batch
        decoding_cfg.preserve_alignments = False
        if hasattr(asr_model, 'joint'):  # if an RNNT model
            decoding_cfg.fused_batch_size = -1
            if not (max_symbols := decoding_cfg.greedy.get("max_symbols")) or max_symbols <= 0:
                decoding_cfg.greedy.max_symbols = 10
        if hasattr(asr_model, "cur_decoder"):
            # hybrid model, explicitly pass decoder type, otherwise it will be set to "rnnt"
            asr_model.change_decoding_strategy(decoding_cfg, decoder_type=asr_model.cur_decoder)
        else:
            asr_model.change_decoding_strategy(decoding_cfg)
    
    asr_model = asr_model.cuda()
    asr_model.eval()
    
    # Set up streaming buffer  
    print("📦 Setting up streaming buffer...")
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=False,
        pad_and_drop_preencoded=False,
    )
    
    # Add audio file to buffer
    print(f"📁 Loading audio file: {audio_file}")
    processed_signal, processed_signal_length, stream_id = streaming_buffer.append_audio_file(
        audio_file, stream_id=-1
    )
    
    # Perform streaming
    print("🎯 Starting streaming inference...")
    start_total = time.time()
    final_streaming_tran, final_offline_tran = perform_streaming(
        asr_model=asr_model,
        streaming_buffer=streaming_buffer,
        compare_vs_offline=True,  # Compare with offline mode
        debug_mode=True,  # Show step-by-step results
        pad_and_drop_preencoded=False,
    )
    total_total = time.time() - start_total
    
    print(f"\n🎤 FINAL RESULTS:")
    print(f"📝 Streaming transcription: '{final_streaming_tran[0] if final_streaming_tran else 'None'}'")
    if final_offline_tran:
        print(f"📝 Offline transcription:   '{final_offline_tran[0] if final_offline_tran else 'None'}'")
    print(f"⏱️  Total time: {total_total*1000:.0f}ms")
    
    # Calculate audio duration for RTF
    import soundfile as sf
    try:
        audio_data, sample_rate = sf.read(audio_file)
        audio_duration = len(audio_data) / sample_rate
        rtf = total_total / audio_duration
        print(f"🚀 Real-time factor: {rtf:.2f}x (lower is better)")
    except Exception as e:
        print(f"⚠️  Could not calculate RTF: {e}")


if __name__ == '__main__':
    main()
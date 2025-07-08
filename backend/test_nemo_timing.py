#!/usr/bin/env python3
"""
Test NeMo FastConformer transcription timing on a sample wav file.
"""

import time
import nemo.collections.asr as nemo_asr

AUDIO_FILE = "test_aws_speech.wav"

MODEL_NAME = "nvidia/stt_en_fastconformer_hybrid_large_streaming_multi"

print(f"Downloading model: {MODEL_NAME}")
asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=MODEL_NAME)

# Optional: change the default latency. Default latency is 1040ms. Supported latencies: {0: 0ms, 1: 80ms, 16: 480ms, 33: 1040ms}.
# Note: These are the worst latency and average latency would be half of these numbers.
asr_model.encoder.set_default_att_context_size([70,13])

# Optional: change the default decoder. Default decoder is Transducer (RNNT). Supported decoders: {ctc, rnnt}.
asr_model.change_decoding_strategy(decoder_type='rnnt')

# Warm up the model with test_audio.wav
print("Warming up model with test_audio.wav...")
warmup_start = time.time()
warmup_output = asr_model.transcribe(["test_audio.wav"])
warmup_elapsed = (time.time() - warmup_start) * 1000  # Convert to milliseconds
print(f"Warm-up transcription: {warmup_output[0].text}")
print(f"Warm-up time: {warmup_elapsed:.2f} milliseconds")

print(f"\nTranscribing file: {AUDIO_FILE}")
start = time.time()
output = asr_model.transcribe([AUDIO_FILE])
elapsed = (time.time() - start) * 1000  # Convert to milliseconds

print(f"\nTranscription: {output[0].text}")
print(f"Elapsed time: {elapsed:.2f} milliseconds") 
#!/bin/bash
# Optimized llama.cpp build for Ryzen 7600X3D with AVX512

cd /home/kevin/orderly/llama.cpp
make clean

# Build with all CPU optimizations
make -j6 \
  GGML_CUDA=1 \
  GGML_OPENBLAS=1 \
  GGML_AVX512=1 \
  GGML_AVX512_VBMI=1 \
  GGML_AVX512_VNNI=1 \
  GGML_FMA=1 \
  GGML_F16C=1 \
  LLAMA_FAST=1 \
  LLAMA_NO_CCACHE=1

echo "Build complete! Now run with:"
echo "taskset -c 0-5 ./build/bin/llama-server --model /path/to/model.gguf --n-gpu-layers 999 --threads 6"
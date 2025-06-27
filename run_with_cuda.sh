#!/bin/bash
# Script to run Python with CUDA support for llama-cpp-python

export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.9/bin:$PATH

source venv/bin/activate
python "$@"
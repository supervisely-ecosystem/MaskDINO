#!/bin/sh
export CUDA_HOME="/usr/local/cuda-12.1"
cd maskdino/modeling/pixel_decoder/ops
sh make.sh
pip3 install .
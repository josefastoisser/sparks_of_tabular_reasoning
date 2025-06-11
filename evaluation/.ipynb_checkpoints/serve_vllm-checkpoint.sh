#!/bin/bash

model_path="$1"

nohup vllm serve "$model_path" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    > output.log 2>&1 &

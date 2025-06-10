#!/bin/bash

nohup vllm serve path/to/model \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    > output.log 2>&1 &

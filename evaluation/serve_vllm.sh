#!/bin/bash

nohup vllm serve Qwen/Qwen2.5-14B-Instruct \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    > output4.log 2>&1 &


#git push  https://josefastoisser:ghp_NF2MWZi9Aw9Xwuf2
YevKjTzsjfpZ4d1p7Ucc@github.com/josefastoisser/sparks_of_tabular_reasoning.git
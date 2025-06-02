# Training

This repository provides two options for training models: using the unsloth or the VERL libarry. Below are instructions for both methods.

## Option 1: Unsloth
For supervised fine tuning run

```
python unsloth_training/SFT.py -- model_name Qwen/Qwen2.5-14B-Instruct
```

For GRPO trainig run

```
python unsloth_training/GRPO.py -- model_name Qwen/Qwen2.5-14B-Instruct
```

## Option 2: VERL

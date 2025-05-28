# Unsloth Training




To run SFT, pick a model, and decide whethe to use the CoT traces or not:

```
python SFT.py --model_name unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit --use_cot 'True'
```

To run GRPO, simply pick a model:

```
python GRPO.py --model_name_or_path unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit
```
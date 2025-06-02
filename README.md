# Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning
### Important Links

📖[Arxiv Paper](https://arxiv.org/abs/2505.00016) |
🤗[Synthetic Data](https://huggingface.co/datasets/jls205/synthetic_cot_traces_clinton/blob/main/cot.csv) |

---

## Introduction

This work reframes the Text-to-SQL task as a pathway for teaching large language models (LLMs) to reason over and manipulate tabular data—moving beyond the traditional focus on query generation. We propose a two-stage framework that leverages SQL supervision to develop transferable table reasoning capabilities:  

1. **Synthetic Chain-of-Thought Traces**:  
   We synthesize detailed chain-of-thought (CoT) traces from real-world SQL queries, providing step-by-step, clause-level supervision that teaches models how to traverse, filter, and aggregate table fields.  

2. **Reinforcement Learning via GRPO**:  
   We introduce a Group Relative Policy Optimization (GRPO) method that connects SQL execution accuracy to generalizable reasoning by encouraging steps that transfer across datasets and logic structures, rather than overfitting to task-specific syntax.

![intro figure](figure/intro_figure.png)


## Empirical Results

Our experiments demonstrate the following:  

- **Enhanced Generalization**:  
  Our approach improves model performance on standard Text-to-SQL datasets like BIRD and achieves substantial gains on reasoning-intensive benchmarks such as CRT-QA and Tablebench.  

- **Significant Improvements with various Models**:  
  - LLaMA-8B (distilled, quantized) achieved a **34% relative increase** in exact match scores on CRT-QA after Text-to-SQL training.  
  - **Qwen-2.5-7B** achieved a **10% gain**, and **Qwen-2.5-14B** recorded a **6% gain**.  

- **Transferable Reasoning**:  
  Our framework shows that SQL can serve not just as a target language for performance but as a scaffold for building robust and interpretable reasoning skills for tabular question answering tasks.

---

![results figure](figure/results_figure.png)


## Getting Started
We rely on unsloth for training and vLLm for evaluation.

```
pip install -r requirements.txt
```

## Training

Refer to [training documentation](training/README.md) for explanations on:  

- Fine-tuning with Synthetic CoT Data.  
- GRPO implementation for reinforcement learning.  

## Evaluation

The evaluation pipeline supports the following tasks:  

1. **Text-to-SQL Benchmarks**: Clinton and BIRD datasets.  
2. **Table-based QA**: CRT-QA and Tablebench datasets.  

Refer to [evaluation documentation](evaluation/README.md).  

## Citation 

```bibtex
@article{stoisser2025sparks,
  title={Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning},
  author={Stoisser, Josefa Lia and Martell, Marc Boubnovski and Fauqueur, Julien},
  journal={arXiv preprint arXiv:2505.00016},
  year={2025}
}
```

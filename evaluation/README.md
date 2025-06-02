# Evals

This is a reusable and easy-to-reproduce evals as seen in the paper.

## Gettinh started

First, download the CRT-QA and BIRD minidev data from:

- https://github.com/zzh-SJTU/CRT-QA
- https://bird-bench.github.io/

and put them into the 'data' folder.

Here we show how to run evals for Qwen/Qwen2.5-14B-Instruct model. After installing, you can run the serving script which log to `output.log`:

```
HF_TOKEN=xyz sh serve_vllm.sh
```

Then the last thing we need is to setup credentials to access the local model and a remote judge:

```bash
export CAND_API_KEY=xyz (huggingface token)
export CAND_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=xyz (OpenAI token)
export JUDGE_BASE_URL=xyz (OpenAI url)
```

## Running evals

We have implemented extensive evaluations. Here are the CLI options to choose from:

```
usage: Universal evaluation harness [-h] --task {bird,clinton,crt_qa,tablebench} --model_path MODEL_PATH [--judge_model JUDGE_MODEL] [--workers WORKERS] [--base_dir BASE_DIR]

options:
  --task {bird,clinton,crt_qa,tablebench}
                        Which benchmark to run
  --model_path MODEL_PATH
                        OpenAI compatible model to evaluate
  --judge_model JUDGE_MODEL
                        OpenAI compatible model to use as judge
  --workers WORKERS     Number of parallel samples to evaluate.
  --base_dir BASE_DIR   The base directory to load data from.
```


Bird:

```
python run_eval.py --task bird --model_path Qwen/Qwen2.5-14B-Instruct
```

Clinton:

```
python run_eval.py --task clinton --model_path Qwen/Qwen2.5-14B-Instruct --judge_model openai_o3_mini
```

CRT QA:

```
python run_eval.py --task crt_qa --model_path Qwen/Qwen2.5-14B-Instruct
```

TableBench:

```
python run_eval.py --task tablebench --model_path Qwen/Qwen2.5-14B-Instruct
```

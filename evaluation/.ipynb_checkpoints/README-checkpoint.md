# Evaluation

This is a reusable and easy-to-reproduce evals as seen in the paper.

## Getting started

First, download the CRT-QA and BIRD minidev data from:

- https://github.com/zzh-SJTU/CRT-QA
- https://bird-bench.github.io/

and put them into the 'data' folder.

Then run the model serving script which log to `output.log`:

```
HF_TOKEN=xyz sh serve_vllm.sh /path/to/model
```


Then the last thing we need is to setup credentials to access the local model and a remote judge:

```bash
export CAND_API_KEY=xyz (huggingface token)
export CAND_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=xyz (OpenAI token)
export JUDGE_BASE_URL=xyz (OpenAI url)
```



## Running evals

```
usage: python run_eval.py --task {bird,clinton,crt_qa,tablebench} --model_path MODEL_PATH [--judge_model JUDGE_MODEL] [--workers WORKERS] [--base_dir BASE_DIR]

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
python run_eval.py --task bird --use_evidence True --model_path /path/to/model
```

Clinton:

```
python run_eval.py --task clinton --model_path /path/to/model --judge_model /path/to/judge_model
```

CRT QA:

```
python run_eval.py --task crt_qa --model_path /path/to/model
```

TableBench:

```
python run_eval.py --task tablebench --model_path /path/to/model
```

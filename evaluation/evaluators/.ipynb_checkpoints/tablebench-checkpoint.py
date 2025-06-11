import re
import logging
from datasets import load_dataset
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from utils import prompts
from utils.helpers import get_EM_score, evaluate_rouge_response_manual, extract_json_answer


class TableBenchEvaluator(BaseEvaluator):
    NAME = "tablebench"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        dataset = load_dataset("Hyeonsieun/Tablebench")
        dataset = (
            dataset["test"]
            .filter(lambda example: example["qtype"] == "FactChecking")
            .shuffle(seed=42)
        )

        return dataset

    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template = self._get_prompt_template()

        conversation = [
            {"content": system, "role": "system"},
            {
                "content": template.format(example["table"], example["question"]),
                "role": "user",
            },
        ]

        response = self.cand(conversation)
        if "llama" in self.args.model_path.lower():
            final_answer=extract_json_answer(response)
        elif "o1" in self.args.model_path.lower():
            final_answer=response.split("Final Answer:")[-1]
        else:
            final_answer = (
                response.split("Final Answer:")[-1]
                if len(response.split("Final Answer:")) > 1
                else ""
            )
            
        ex = re.findall(r"\d+\.\d+|\d+", example["answer"])
        ex = str(ex[0]) if len(ex) > 0 else example["answer"]

        rouge_manual=evaluate_rouge_response_manual( ex, final_answer)

        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {'rouge_score': rouge_manual},
        }
        
    def _get_prompt_template(self):
        from utils import prompts

        prompt_mapping = {
            "qwen": (
                prompts.TAB_QWEN_SYSTEM,
                prompts.TAB_QWEN_TEMPLATE,
            ),
            "llama": (
                prompts.TAB_LLAMA_SYSTEM,
                prompts.TAB_LLAMA_TEMPLATE,
            ),
            "o1": (
                prompts.TAB_o1_SYSTEM,
                prompts.TAB_o1_TEMPLATE,
            ),
        }

        model_format = 'qwen'
        if "llama" in self.args.model_path.lower():
            model_format = "llama"
        if "o1" in self.args.model_path.lower():
            model_format = "o1"

        system, template = prompt_mapping[model_format]
        return system, template

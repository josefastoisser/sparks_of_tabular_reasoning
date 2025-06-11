import os
import json
import random
import pandas as pd
from typing import Dict, Any
from argparse import ArgumentParser
from .base import BaseEvaluator
from utils.helpers import get_EM_score, extract_json_answer
import evaluate
import logging


class CRTQAEvaluator(BaseEvaluator):
    NAME = "crt_qa"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        csv_folder_path = f"{self.args.base_dir}/CRT-QA/all_csv_new"

        with open(f"{self.args.base_dir}/CRT-QA/dataset.json") as file:
            dataset_json = json.load(file)

        all_tasks = []

        for file_name in dataset_json:
            csv_file_path = f"{csv_folder_path}/{file_name}"
            if os.path.exists(csv_file_path):
                csv_data = pd.read_csv(csv_file_path, sep="#")
                for question in dataset_json[file_name]:
                    question_text = question["Question name"]
                    answer = question["Answer"]
                    title = question["Tittle"]
                    all_tasks.append((title, question_text, csv_data, answer))

        random.Random(42).shuffle(all_tasks)

        return all_tasks

    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template = self._get_prompt_template()
        title, question_text, csv_data, answer = example

        conversation = [
            {"content": system, "role": "system"},
            {
                "content": template.format(
                    title,
                    csv_data.to_json(orient="records", lines=False),
                    question_text,
                ),
                "role": "user",
            },
        ]
        response = self.cand(conversation)

        if "llama" in self.args.model_path.lower():
            final_answer=extract_json_answer(response)
        else:
            final_answer = response.split("</think>")[-1]
        score_em_manual = get_EM_score(answer, final_answer)
        score_in = 1 if str(answer).lower() in str(final_answer).lower() else 0

        return {
            "question": question_text,
            "llm": response,
            "answer": answer,
            "scores": {'EM_manual': score_em_manual},
        }

    def _get_prompt_template(self):
        from utils import prompts

        prompt_mapping = {
            "qwen": (
                prompts.CRT_QWEN_SYSTEM,
                prompts.CRT_QWEN_TEMPLATE,
            ),
            "llama": (
                prompts.CRT_LLAMA_SYSTEM,
                prompts.CRT_LLAMA_TEMPLATE,
            ),
            "o1": (
                prompts.CRT_o1_SYSTEM,
                prompts.CRT_o1_TEMPLATE,
            ),
        }

        model_format = 'qwen'
        if "llama" in self.args.model_path.lower():
            model_format = "llama"
        if "o1" in self.args.model_path.lower():
            model_format = "o1"

        system, template = prompt_mapping[model_format]
        return system, template
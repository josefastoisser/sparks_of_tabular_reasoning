import pandas as pd
from typing import Dict, Any
from datasets import Dataset
from argparse import ArgumentParser
from langchain_community.utilities import SQLDatabase

from .base import BaseEvaluator
from utils.helpers import execute_sqlite, extract_xml_answer_sql
from utils import prompts

class BirdEvaluator(BaseEvaluator):
    NAME = "bird"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        p.add_argument(
            "--use_evidence",
            type=bool,
            default=False,
            required=True,
            help="Use evidence",
        )

    def load_data(self) -> Dataset:
        """
        1. Read the json exported from MiniDev.
        2. Build a hf-Dataset.
        3. Collect table-info for every db_id once, not per row.
        4. Attach the corresponding context string to every example.
        """
        # A) read the raw json ------------------------------------------------
        json_path = f"{self.args.base_dir}/MINIDEV/mini_dev_sqlite.json"
        df = pd.read_json(json_path)
        ds = Dataset.from_pandas(df)

        # B) pre-compute every DB context only once --------------------------
        self.context_per_db: Dict[str, str] = {}
        for db_id in set(ds["db_id"]):
            sqlite_file = (
                f"{self.args.base_dir}/MINIDEV/" f"dev_databases/{db_id}/{db_id}.sqlite"
            )
            db = SQLDatabase.from_uri(
                f"sqlite:///{sqlite_file}", sample_rows_in_table_info=0
            )
            self.context_per_db[db_id] = db.get_context()["table_info"]

        # C) small helper to enrich a row with its context -------------------
        def attach_context(example):
            example["context"] = self.context_per_db[example["db_id"]]
            return example

        # D) final dataset pipeline ------------------------------------------
        ds = (
            ds.rename_columns({"SQL": "answer"})  # align with common field-name
            .shuffle(seed=42)
            .map(attach_context)
        )
        return ds

    def evaluate_one(self, example) -> Dict[str, Any]:
        system, template, evidence = self._get_prompt_template()

        if self.args.use_evidence:
            conversation = [
                {"content": system, "role": "system"},
                {
                    "content": prompts.BIRD_QWEN_TEMPLATE_EVIDENCE.format(
                        example["context"], example["evidence"], example["question"]
                    ),
                    "role": "user",
                },
            ]
        else:
            conversation = [
                {"content": system, "role": "system"},
                {
                    "content": template.format(example["context"], example["question"]),
                    "role": "user",
                },
            ]

        response = self.cand(conversation)
        pred_query = extract_xml_answer_sql(response)

        sqlite_file = (
            f"{self.args.base_dir}/MINIDEV/"
            f"dev_databases/{example['db_id']}/{example['db_id']}.sqlite"
        )

        try:
            correct_results = execute_sqlite(example["answer"], sqlite_file)
        except Exception as e:
            correct_results = f"Error executing correct SQL: {e}"
            return 
        if correct_results == None or len(correct_results) == 0:
            correct_results = "No results returned for the correct SQL query."
            return

        try:
            pred_results = execute_sqlite(pred_query, sqlite_file)
        except Exception as e:
            pred_results = f"Error executing predicted SQL: {e}"

        results_match = set(pred_results) == set(correct_results)

        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {"execution_accuracy": results_match},
        }

    def _get_prompt_template(self):
        from utils import prompts

        prompt_mapping = {
            "qwen": (
                prompts.BIRD_SYSTEM,
                prompts.BIRD_QWEN_TEMPLATE,
                prompts.BIRD_QWEN_TEMPLATE_EVIDENCE,
            ),
            "llama": (
                prompts.BIRD_SYSTEM,
                prompts.BIRD_LLAMA_TEMPLATE,
                prompts.BIRD_LLAMA_TEMPLATE_EVIDENCE,
            ),
            "o1": (
                'You are a helpful assistant.',
                prompts.BIRD_QWEN_TEMPLATE,
                prompts.BIRD_QWEN_TEMPLATE_EVIDENCE,
            ),
        }

        model_format = "qwen"
        if "llama" in self.args.model_path.lower():
            model_format = "llama"

        system, template, evidence = prompt_mapping[model_format]
        return system, template, evidence


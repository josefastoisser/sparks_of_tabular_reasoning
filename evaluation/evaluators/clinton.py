from typing import Dict, Any
from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict, interleave_datasets, Dataset
from .base import BaseEvaluator
from utils.helpers import (
    extract_xml_answer_sql,
    evaluate_generated_response,
)
from langchain_openai import ChatOpenAI


class ClintonEvaluator(BaseEvaluator):
    NAME = "clinton"

    @classmethod
    def add_args(cls, p: ArgumentParser):
        pass

    def load_data(self):
        # load SQL create context
        scc_path = "b-mc2/sql-create-context"
        dataset_scc_train = load_dataset(scc_path, split="train[:80%]")
        dataset_scc_test = load_dataset(scc_path, split="train[-20%:-10%]")
        dataset_scc_val = load_dataset(scc_path, split="train[-10%:]")

        # load text to sql test+val splits
        tts_path = "Clinton/Text-to-sql-v1"
        rename_columns = {
            "instruction": "question",
            "input": "context",
            "response": "answer",
        }
        dataset_tts_train = load_dataset(tts_path, split="train[:80%]")
        dataset_tts_train = dataset_tts_train.remove_columns(["source", "text"])
        dataset_tts_train = dataset_tts_train.rename_columns(rename_columns)

        dataset_tts_test = load_dataset(tts_path, split="train[-20%:-10%]")
        dataset_tts_test = dataset_tts_test.remove_columns(["source", "text"])
        dataset_tts_test = dataset_tts_test.rename_columns(rename_columns)

        dataset_tts_val = load_dataset(tts_path, split="train[-10%:]")
        dataset_tts_val = dataset_tts_val.remove_columns(["source", "text"])
        dataset_tts_val = dataset_tts_val.rename_columns(rename_columns)

        # load know sql
        know_path = "knowrohit07/know_sql"
        dataset_ks_train = load_dataset(know_path, split="validation[:80%]")
        dataset_ks_test = load_dataset(know_path, split="validation[-20%:-10%]")
        dataset_ks_val = load_dataset(know_path, split="validation[-10%:]")

        dataset = DatasetDict(
            {
                "train": interleave_datasets(
                    [dataset_scc_train, dataset_tts_train, dataset_ks_train]
                ),
                "test": interleave_datasets(
                    [dataset_scc_test, dataset_tts_test, dataset_ks_test]
                ),
                "validation": interleave_datasets(
                    [dataset_scc_val, dataset_tts_val, dataset_ks_val]
                ),
            }
        )

        dataset_1_test = dataset["test"].shuffle(seed=42)[:1000]
        dataset_1_test = Dataset.from_dict(dataset_1_test)

        return dataset_1_test

    def evaluate_one(self, example) -> Dict[str, Any]:
        from utils import prompts

        template = prompts.CLINTON_QWEN_TEMPLATE
        if "llama" in self.args.model_path.lower():
            template = prompts.CLINTON_LLAMA_TEMPLATE


        conversation = [
            {"content": 'You are a helpful assistant' if 'o1' in self.args.model_path.lower() else prompts.CLINTON_SYSTEM, "role": "system"},
            {
                "content": template.format(example["context"], example["question"]),
                "role": "user",
            },
        ]


        response = self.cand(conversation)
        pred_query = extract_xml_answer_sql(response)


        sql_eval = [
            (
                "system",
                f"""
                    You are SQL expert and your task is to evaluate if if the predicted SQL query is correct 
                    based on the Schema and the correct SQL query. If no SQL query was found then the answer is Wrong. The query is considered correct even if the only mistakes are in letter casing (uppercase vs lowercase).
                    Schema: {example['context']}
                    Predicted query: {pred_query}
                    correct SQL query: {example['answer']}

                    Return ONLY "Correct" or "Wrong"
                        """,
            ),
            (pred_query),
        ]

        score = evaluate_generated_response(self.judge.chat, sql_eval)

        return {
            "question": example["question"],
            "llm": response,
            "answer": example["answer"],
            "scores": {"llm_accuracy": score},
        }

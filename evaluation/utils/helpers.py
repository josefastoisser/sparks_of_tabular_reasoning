import re
import sys
import json
import time
import sqlite3
import pandas as pd
from typing import Optional, List
from collections import OrderedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from collections import defaultdict
from typing import List, Dict
from math import exp, log


def execute_sqlite(
    sql: str, db_path: str, timeout_s: float = 10.0, instruction_chunk: int = 1_000
):
    start = time.time()

    def _progress_handler():
        if time.time() - start > timeout_s:
            return 1
        return 0

    uri = f"file:{db_path}"
    with sqlite3.connect(uri, uri=True) as con:
        con.set_progress_handler(_progress_handler, instruction_chunk)
        cur = con.cursor()
        return cur.execute(sql).fetchall()



def extract_xml_answer_sql(text: str) -> str:
    """
    Extract the last SQL query from a response that contains ```sql and ``` markers.
    Falls back to xml extraction if SQL markers aren't found.
    """
    if "```sql" in text:
        sql_blocks = text.split("```sql")

        if len(sql_blocks) > 1:
            last_sql_part = sql_blocks[-1]
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]
                return query.strip()
    print("No SQL query found in response")
    return " "

class sql_evaluator(BaseModel):
    grade: Optional[str] = Field(default=None, description="The sql evaluation")


parser_eval = PydanticOutputParser(pydantic_object=sql_evaluator)


def evaluate_generated_response(llm: ChatOpenAI, sql_eval: List[str]):

    eval_prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={
            "format_instructions": parser_eval.get_format_instructions()
        },
    )

    eval_chain = eval_prompt | llm | parser_eval
    eval_res = eval_chain.invoke(sql_eval)

    try:
        return 1 if eval_res.grade.lower() == "correct" else 0
    except AttributeError:
        return 0


def get_EM_score(gold_answer, llm_answer):
    def normalize(text):
        text = str(text).lower()
        punctuation = """!()-[]{};:'"\,<>./?@#$%^&*_~"""
        normalized_text = "".join(char for char in text if char not in punctuation)
        return " ".join(normalized_text.split()).strip()

    normalized_gold = normalize(gold_answer)
    normalized_llm = normalize(llm_answer)

    try:
        gold_float = float(normalized_gold)
        llm_float = float(normalized_llm)
        return 1 if round(gold_float, 2) == round(llm_float, 2) else 0
    except ValueError:
        return 1 if normalized_gold == normalized_llm else 0

def extract_json_answer(text):
    # Use a regular expression to extract the answer part
    match =  re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        # Extract the JSON part
        json_part = match.group(1)
        # Convert the JSON string into a Python dict
        try:
            answer_data = json.loads(json_part)
            return answer_data.get("answer")
        except Exception as e:  # This will catch any error
            print(f"Error processing answer: {e}")
            return None
    else:
        return None
        
def evaluate_rouge_response_manual(reference,prediction):
    def get_lcs_length(x, y):
        # Create LCS matrix
        m, n = len(x), len(y)
        lcs = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])
        
        return lcs[m][n]

    # Convert inputs to lowercase and split into words
    prediction=str(prediction)
    pred_words = prediction.lower().split()
    ref_words = reference.lower().split()
    
    # Get length of LCS
    lcs_length = get_lcs_length(pred_words, ref_words)
    
    # Calculate precision, recall, and f1
    if len(pred_words) == 0:
        precision = 0.0
    else:
        precision = lcs_length / len(pred_words)
        
    if len(ref_words) == 0:
        recall = 0.0
    else:
        recall = lcs_length / len(ref_words)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return f1



def calculate_average_scores(results: List[Dict[str, Dict[str, float]]]) -> str:
    totals = defaultdict(int)
    counts = defaultdict(int)

    # Iterate through all the results
    for res in results:
        if res:
            
            # Aggregate scores into totals and counts
            for subject, score in res["scores"].items():
                totals[subject] += score
                counts[subject] += 1

    # Calculate average scores rounded to 3 decimal places
    average_scores = {subject: round(totals[subject] / counts[subject], 3) for subject in totals}
    
    # Create a final formatted string of the average scores
    final_string = ', '.join(f'{subject}: {avg}' for subject, avg in average_scores.items())
    
    return final_string


def calculate_average_scores_dict(results: List[Dict[str, Dict[str, float]]]) -> str:
    totals = defaultdict(int)
    counts = defaultdict(int)

    # Iterate through all the results
    for res in results:
        if res:
            # Aggregate scores into totals and counts
            for subject, score in res["scores"].items():
                totals[subject] += score
                counts[subject] += 1

    # Calculate average scores rounded to 3 decimal places
    average_scores = {subject: round(totals[subject] / counts[subject], 3) for subject in totals}
        
    return average_scores
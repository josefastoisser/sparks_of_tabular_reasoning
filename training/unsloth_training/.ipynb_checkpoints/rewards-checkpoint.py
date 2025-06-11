import os
import numpy as np
import re
import random
import wandb
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from sqlglot import exp, parse_one
import json
import difflib
from typing import List, Set, Optional, Dict
from pydantic import BaseModel, Field
from collections import defaultdict


np.random.seed(88)

llm = ChatOpenAI(model='openai_o3_mini', model_kwargs={'temperature':1 }, base_url=os.getenv("OPENAI_URL"), api_key=os.getenv("OPENAI_API_KEY"))


def normalize_sql_query(query: str) -> str:
    """
    Normalize an SQL query string by applying transformations to its expression tree.

    :param query: The SQL query string to normalize.
    :return: The normalized SQL query string, or the original query if an error occurs.
    """
    try:
        expression_tree = parse_one(query)

        def transformer(node):
            if isinstance(node, exp.Column) and node.name == "a":
                return parse_one("FUN(a)")
            return node

        transformed_tree = expression_tree.transform(transformer)

        return transformed_tree.sql()

    except Exception as e:
        return query


def extract_xml_answer(text: str) -> str:
    """
    Extract the last SQL query from a response that contains ```sql and ``` markers.
    Returns empty string if no SQL markers are found.
    """
    if "```sql" in text:
        sql_parts = text.split("```sql")[1:]
        
        if sql_parts:
            last_sql_part = sql_parts[-1]
            if "```" in last_sql_part:
                query = last_sql_part.split("```")[0]
                
                return normalize_sql_query(query.strip())
    
    print("No SQL query found in response")
    return ""


############### STRING MATCHING ####################


def string_matching_reward(prompts, completions, answer, **kwargs) -> list:
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [r for r in responses]
    rewards = []
    
    for pred_answer, true_query in zip(extracted_responses, answer):
        try:
            pred_query=extract_xml_answer(pred_answer)
            similarity = difflib.SequenceMatcher(None, pred_query, true_query).ratio()

            reward = similarity

            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("Tdring Matching Reward:", reward)
            
            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating string match: {e}")
            rewards.append(0.0)
    return rewards





# ################ LLM scoring classes judge ################

class QueryEvaluation(BaseModel):
    grades: list[str] = Field(
        description="List of grades for each query: 'Very bad', 'Bad', 'Average', 'Above average', 'Good', or 'Excellent'",
        min_items=1
    )

def llm_scoring_classes_judge_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Reward function that uses LLM to grade SQL queries with categorical ratings.
    Returns list of rewards based on grades.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    prompt_keys = [str(prompt) for prompt in prompts]
    
    grade_rewards = {
        'Excellent': 1.0,
        'Good': 0.8,
        'Above average': 0.6,
        'Bad': 0.2,
        'Very bad': 0.0
    }
    
    query_groups = {}
    for i, (prompt_key, pred_query, true_query) in enumerate(zip(prompt_keys, extracted_responses, answer)):
        if prompt_key not in query_groups:
            query_groups[prompt_key] = []
        query_groups[prompt_key].append({
            'index': i,
            'pred_query': pred_query,
            'true_query': true_query
        })
    
    parser = PydanticOutputParser(pydantic_object=QueryEvaluation)
    
    eval_template = """Compare these SQL queries to the correct query and grade each one as: 'Very bad', 'Bad', 'Above average', 'Good', or 'Excellent'.
    This is the following grading system, use the correct query as reference :
    
    - Correct Query: {true_query}
    
    1. Excellent: this is only given when the SQL query is perfect and matches {true_query}
    2. Good: This is when there is a grammar mistake in the query
    3. Above average: This is when the query is mostly correct but gets a logical step wrong in the query
    4. Bad: Makes more than one mistake in the query 
    5. Very bad: does not produce a query or varies significantly from the correct query



    Queries to grade:
    {queries_to_rank}

    {format_instructions}

    Return ONLY the JSON with grades list."""

    rewards = [0.0] * len(completions)
    
    for prompt, group in query_groups.items():
        try:
            num_queries = len(group)
            queries_text = "\n\n".join([f"Query {i+1}:\n{q['pred_query']}" 
                                      for i, q in enumerate(group)])
            
            eval_prompt = PromptTemplate(
                template=eval_template,
                input_variables=["queries_to_rank", "true_query", "num_queries"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            eval_chain = eval_prompt | llm | parser
            
            eval_result = eval_chain.invoke({
                "queries_to_rank": queries_text,
                "true_query": group[0]['true_query'],
                "num_queries": num_queries
            })
            
            for i, grade in enumerate(eval_result.grades):
                index = group[i]['index']
                rewards[index] = grade_rewards[grade]
                
                print(f"\nQuery {i+1}: {group[i]['pred_query']}")
                print(f"Grade: {grade}")
                print(f"LLM Judge Reward: {rewards[index]}")
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            for query_info in group:
                rewards[query_info['index']] = 0.0
    
    return rewards



# #################### LLM ex reward ####################

class SQLEvaluator_llm_exe_reward(BaseModel):
    scores: list[int] = Field(
        description="List of Number of Orthographic elements needed to change from Predicted query to True Query"
    )

def llm_exe_reward(prompts, completions, answer,context, **kwargs) -> list[float]:
    """
    Reward function that uses LLM to score SQL queries on a scale of 0 to 2.
    Returns list of scores where better queries get higher scores.
    """

    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    prompt_keys = [str(prompt) for prompt in prompts]
    
    query_groups = {}
    for i, (prompt_key, pred_query, true_query, q_context) in enumerate(zip(prompt_keys, extracted_responses, answer, context)):
        if prompt_key not in query_groups:
            query_groups[prompt_key] = []
        query_groups[prompt_key].append({
            'index': i,
            'pred_query': pred_query,
            'true_query': true_query,
            'context': q_context
        })
    
    parser = PydanticOutputParser(pydantic_object=SQLEvaluator_llm_exe_reward)
    
    eval_template = """You are SQL expert. Count how many changes you need to make to get the following Predicted queries correct.
    
Predicted Queries (one per line): 
{queries_to_rank}

For reference, use this Schema: {context}
    
Here are the correct queries: 
{true_query}
    
You should count the number of Orthographic elements you need to change from the predicted queries to the correct queries.
    
ONLY RETURN a JSON object with a single 'scores' field containing a list of {num_queries} numbers reflecting the number of changes needed for each predicted query."""

    rewards = [0.0] * len(completions)
    
    for prompt, group in query_groups.items():
        try:
            num_queries = len(group)
            queries_text = "\n\n".join([f"Query {i+1}:\n{q['pred_query']}" 
                                      for i, q in enumerate(group)])
            
            eval_prompt = PromptTemplate(
                template=eval_template,
                input_variables=["queries_to_rank", "true_query", "num_queries", "context"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            eval_chain = eval_prompt | llm | parser
            
            eval_result = eval_chain.invoke({
                "queries_to_rank": queries_text,
                "true_query": group[0]['true_query'],
                "num_queries": num_queries,
                'context': group[0]['context']
            })
            print(f"\n true query:{group[0]['true_query']}")

            
            for i, score in enumerate(eval_result.scores):
                index = group[i]['index']
                rewards[index] = 1 / (int(score) + 1)
                
                print(f"\nQuery {i+1}: {group[i]['pred_query']}")
                print(f"Exe LLM Score: {1 / (int(score) + 1)}")
                
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            for query_info in group:
                rewards[query_info['index']] = 0.0
    
    return rewards


################## COMPONENT MATHCING ###############



class SQLSimilarityChecker:
    def __init__(self):
        self.keywords = {
            "main_body": ["SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", 
                         "EXISTS", "IS", "NULL", "IIF", "CASE", "CASE WHEN"],
            "join": ["INNER JOIN", "LEFT JOIN", "ON", "AS"],
            "clause": ["BETWEEN", "LIKE", "LIMIT", "ORDER BY", "ASC", "DESC", 
                      "GROUP BY", "HAVING", "UNION", "ALL", "EXCEPT", "PARTITION BY"],
            "aggregation": ["AVG", "COUNT", "MAX", "MIN", "ROUND", "SUM"],
            "scalar": ["ABS", "LENGTH", "STRFTIME", "JULIADAY", "NOW", "CAST", 
                      "SUBSTR", "INSTR"],
        }
        
    def _extract_comma_separated(self, sql, start_pos):
        """Extract comma-separated items until next keyword"""
        next_keyword_pos = len(sql)
        sql_upper = sql.upper()
        for category_keywords in self.keywords.values():
            for keyword in category_keywords:
                pos = sql_upper.find(' '+keyword+' ', start_pos)
                if pos != -1 and pos < next_keyword_pos:
                    next_keyword_pos = pos
        
        content = sql[start_pos:next_keyword_pos].strip()
        items = [item.strip() for item in content.split(',')]
        return tuple(filter(None, items))  # Return tuple instead of set
        
    def _extract_comparison(self, sql, pos):
        """Extract comparison expression"""
        next_pos = len(sql)
        sql_upper = sql.upper()
        for category_keywords in self.keywords.values():
            for keyword in category_keywords:
                keyword_pos = sql_upper.find(keyword, pos + 1)
                if keyword_pos != -1 and keyword_pos < next_pos:
                    next_pos = keyword_pos
        
        return sql[pos:next_pos].strip()
        
    def _extract_case_statement(self, sql, pos):
        """Extract complete CASE statement"""
        sql_upper = sql.upper()
        end_pos = sql_upper.find("END", pos)
        if end_pos != -1:
            return sql[pos:end_pos + 3].strip()
        return ""
        
    def _extract_component(self, sql, pos, keyword):
        """Extract meaningful component after keyword"""
        keyword_upper = keyword.upper()
        
        def find_next_keyword_pos(start_pos):
            next_pos = len(sql)
            for cat_keywords in self.keywords.values():
                for k in cat_keywords:
                    spaced_k = f" {k} "
                    k_pos = sql.upper().find(spaced_k, start_pos)
                    if k_pos != -1 and k_pos < next_pos:
                        next_pos = k_pos
                    for special_char in ['(', ')', ',', '\n']:
                        spaced_k_special = f" {k}{special_char}"
                        k_pos = sql.upper().find(spaced_k_special, start_pos)
                        if k_pos != -1 and k_pos < next_pos:
                            next_pos = k_pos
            return next_pos
    
        if keyword_upper in ["SELECT", "GROUP BY", "ORDER BY", "WHERE"]:
            return self._extract_comma_separated(sql, pos + len(keyword))
            
        elif keyword_upper in ["CASE", "CASE WHEN"]:
            return self._extract_case_statement(sql, pos)

        else:
            next_pos = find_next_keyword_pos(pos + len(keyword))
            return sql[pos + len(keyword):next_pos].strip()

    def parse_sql(self, sql_query):
        """Parse SQL query into components by keyword category"""
        components = defaultdict(list)
        
        sql_query = f" {sql_query} "
        sql_upper = sql_query.upper()
        
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                spaced_keyword = f" {keyword} "
                start_pos = 0
                
                while True:
                    pos = sql_upper.find(spaced_keyword, start_pos)
                    if pos == -1:
                        for special_char in ['(', ')', ',', '\n']:
                            spaced_keyword_special = f" {keyword}{special_char}"
                            pos = sql_upper.find(spaced_keyword_special, start_pos)
                            if pos != -1:
                                break
                        if pos == -1:
                            break
                    
                    actual_pos = pos + 1
                    component = self._extract_component(sql_query, actual_pos, keyword)
                    if component:
                        components[category].append((keyword, component))
                    
                    start_pos = pos + len(keyword)
        
        return components

    def _calculate_f1(self, list1, list2):
        """Calculate F1 score between two lists of components"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
            
        set1 = {str(item) for item in list1}
        set2 = {str(item) for item in list2}
            
        intersection = len(set1.intersection(set2))
        precision = intersection / len(set1)
        recall = intersection / len(set2)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

    def calculate_similarity(self, sql1, sql2):
        """Calculate similarity between two SQL queries"""
        # Replace AND/OR with commas
        sql1 = sql1.replace(" AND ", " , ").replace(" OR ", " , ")
        sql2 = sql2.replace(" AND ", " , ").replace(" OR ", " , ")
        sql1 = sql1.replace(" and ", " , ").replace(" or ", " , ")
        sql2 = sql2.replace(" and ", " , ").replace(" or ", " , ")
        
        components1 = self.parse_sql(sql1)
        components2 = self.parse_sql(sql2)
        
        all_components1 = []
        all_components2 = []
        
        for category_components in components1.values():
            all_components1.extend(category_components)
            
        for category_components in components2.values():
            all_components2.extend(category_components)

        keyword_scores = {}
        all_keywords = set([comp[0] for comp in all_components1] + [comp[0] for comp in all_components2])
        
        for keyword in all_keywords:
            components_1 = [comp[1] for comp in all_components1 if comp[0] == keyword ]
            components_2 = [comp[1] for comp in all_components2 if comp[0] == keyword]
            
            score = self._calculate_f1(components_1[0] if components_1 !=[] else components_1, components_2[0] if components_2 !=[] else components_2)
            keyword_scores[keyword] = score
            
        
        final_score = sum(keyword_scores.values()) / len(keyword_scores) if keyword_scores else 0.0
        return final_score

def component_matching_new(prompts, completions, answer, **kwargs) -> list[float]:
    checker = SQLSimilarityChecker()
    
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = []
    
    for pred_query, true_query in zip(extracted_responses, answer):
        try:
            similarity = checker.calculate_similarity(pred_query, true_query)
            
            reward = similarity  
            
            print("\nPredicted Query:", pred_query)
            print("True Query:", true_query)
            print("Similarity:", similarity)
            print("Component Reward:", reward)

            rewards.append(reward)
        
        except Exception as e:
            print(f"Error calculating string match: {e}")
            rewards.append(0.0)

    return rewards
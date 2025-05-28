

BIRD_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: 
"""

BIRD_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.

### Response:"""

BIRD_QWEN_TEMPLATE_EVIDENCE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {}.
You can use this external knowledge for help: {}.
Write a SQLite query for the following task: {}.

### Response:"""


BIRD_LLAMA_TEMPLATE = """\
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags. 
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.
Assistant: <think> """

BIRD_LLAMA_TEMPLATE_EVIDENCE = """\
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags. 
Here is the relevant table info: {}.
You can use this external knowledge for help: {}.
Write a SQLite query for the following task: {}.
Assistant: <think> """








CLINTON_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """


CLINTON_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.

### Response:"""

CLINTON_LLAMA_TEMPLATE = """\
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags. 
Here is the relevant table info: {}.
Write a SQLite query for the following task: {}.
Assistant: <think> """









CRT_QWEN_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process is enclosed within <think> </think> tags, respectively, i.e., <think> reasoning process here </think> answer here. User: """

CRT_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a tabular reasoning expert. Read the table below regarding {} to answer the following question. Encose you reasoning process in <think> and </think> tags and provide a clear and concise final answer after the </think> tag, if possible in only one word or one number rounded to two decimals.

### Input:
Table: {}.
Question: {}.

### Response:"""


CRT_LLAMA_SYSTEM="""Provide a clear and concise final answer, if possible in only one word or one number rounded to two decimals in json format like this: ```json {optional other arguments , "answer": the final answer} ```."""

CRT_LLAMA_TEMPLATE = """You are a tabular reasoning expert. Read the table below regarding {} to answer the following question.
Table: {}
Question: {}
Assistant:  """

CRT_o1_SYSTEM="""Provide a clear and concise final answer, if possible in only one word or one number rounded to two decimals. DO NOT INCLUDE ANY REASONING OR EXPLANATIONS."""

CRT_o1_TEMPLATE = """You are a tabular reasoning expert. Read the table below regarding {} to answer the following question.
Table: {}
Question: {}
Assistant:  """







TAB_QWEN_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it,
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The answer should follow the format below:
<think> reasoning process here </think> Final Answer: AnswerName1, AnswerName2...\nEnsure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. User: """

TAB_QWEN_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a table analyst. Your task is to answer questions based on the table content. 

### Input:
Read the table below in JSON format: {}.
Answer the following question: {}.

### Response:"""


TAB_LLAMA_SYSTEM="""Provide a clear and concise final answer in json format like this: ```json {optional other arguments , "answer":  AnswerName1, AnswerName2} ```. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation."""


TAB_LLAMA_TEMPLATE = """You are a table analyst. Your task is to answer questions based on the table content. Provide a clear and concise final answer.
Read the table below in JSON format: {}.
Answer the following question: {}.
Assistant: """


TAB_o1_SYSTEM = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The answer should follow the format below:
Final Answer: AnswerName1, AnswerName2...\nEnsure the final answer format is the last output line and can only be in the "Final Answer: AnswerName1, AnswerName2..." form, no other form. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation. User: """

TAB_o1_TEMPLATE = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a table analyst. Your task is to answer questions based on the table content.

### Input:
Read the table below in JSON format: {}.
Answer the following question: {}.

### Response:"""



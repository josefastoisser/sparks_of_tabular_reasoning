from unsloth import FastLanguageModel, FastModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
import pandas as pd
import wandb
import argparse
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='SFT Training Arguments')
    parser.add_argument('--model_name', type=str, default="unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
                        help='Name of the base model')
    parser.add_argument('--use_cot', type=str, default='True',
                        help='Whether CoT is used')
    parser.add_argument('--checkpoint_dir', type=str, default='model_checkpoints_SFT',
                        help='Directory to save checkpoints')
    parser.add_argument('--final_model_dir', type=str, default='final_models_SFT',
                        help='Directory to save final model')
    return parser.parse_args()


args = parse_args()

max_seq_length = 2048

wandb.login() 
wandb.init(
    project="SFT",
    name=args.model_name
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

model = FastModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

EOS_TOKEN = tokenizer.eos_token

# Replace with your dataset repo ID and the name of the file
data = load_dataset("jls205/synthetic_cot_traces_clinton", data_files="cot.csv")['train'].shuffle(seed=42)


train_size = int(0.99 * len(data))
train_dataset = data.select(range(train_size))
val_dataset = data.select(range(train_size, len(data)))


sql_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction: 
You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags.

### Input:
Here is the relevant table info: {table_info}.
Write a SQLite query for the following task: {question}.

### Response:
{answer}"""


if 'llama' in args.model_name.lower():

    sql_prompt="""You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to run. Enclose the final sql query within "```sql" and "```" tags. 
    Here is the relevant table info: {table_info}.
    Write a SQLite query for the following task: {question}.
    Assistant: {answer}"""

def formatting_prompts_func(examples):
    question = examples['question']
    table_info = examples['context']
    query = "```sql " + examples['answer'] + " ```"

    if args.use_cot == 'True':
        thinking = examples['thinking'].replace('<thinking>', '').replace('</thinking>', '')
        thinking = '<think> ' + thinking + ' </think>'
        answer = thinking + query
    else:
        answer = query

    text = sql_prompt.format(table_info=table_info, question=question, answer=answer) + EOS_TOKEN
    return {"text": text}

train_dataset = train_dataset.map(formatting_prompts_func, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(formatting_prompts_func, remove_columns=val_dataset.column_names)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        eval_steps=10,
        save_strategy="steps",
        save_steps=300,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.checkpoint_dir,
        report_to="wandb",
    ),
)

trainer.train()

model.save_pretrained(args.final_model_dir)
tokenizer.save_pretrained(args.final_model_dir)

wandb.finish()


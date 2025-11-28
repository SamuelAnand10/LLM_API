# finetune_lora_qlora.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Choose a base model you have access to (license/availability)
MODEL = os.environ.get("BASE_MODEL", "tiiuae/falcon-7b-instruct")
DATA_FILE = os.environ.get("DATA_FILE", "assistant_data_cleaned_merged.jsonl")
OUTPUT = os.environ.get("OUTPUT_DIR", "llm-assistant-lora")

print("Using model:", MODEL)
print("Data file:", DATA_FILE)
print("Output dir:", OUTPUT)

# 1) Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

# 2) Load dataset (expects JSONL with {"prompt":"...","response":"..."} per line)
ds = load_dataset("json", data_files=DATA_FILE, split="train")

def format_example(ex):
    prompt = ex.get("prompt","")
    response = ex.get("response","")
    text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}\n"
    return {"text": text}

ds = ds.map(format_example, remove_columns=ds.column_names)

# 3) Tokenize
def tokenize_fn(batch):
    encoding = tokenizer(
        batch["text"],
        truncation=True,
        max_length=1024
    )
    
    # Add labels for training (causal LM expects labels=input_ids)
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# 4) Load quantized model (4-bit) and prepare for LoRA
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    low_cpu_mem_usage=True
)

model = prepare_model_for_kbit_training(model)

# 5) LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 6) Training args
training_args = TrainingArguments(
    output_dir=OUTPUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_steps=2000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=3,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=default_data_collator,
)

if __name__ == "__main__":
    trainer.train()
    model.save_pretrained(OUTPUT)
    tokenizer.save_pretrained(OUTPUT)
    print("Training finished. Saved to", OUTPUT)

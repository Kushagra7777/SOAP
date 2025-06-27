# # ---------------------------------------------------------
# # ✅ Falcon-1B LoRA fine-tuning — FINAL, STABLE VERSION for 8GB RTX 4060.
# # 2025-06-23 — with all fixes: PEFT, grad checkpoint, input grads, correct masking.
# # ---------------------------------------------------------

# import os
# import json
# from datasets import Dataset
# from huggingface_hub import login
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     TrainingArguments,
#     Trainer,
#     BitsAndBytesConfig,
#     default_data_collator
# )
# from peft import LoraConfig, get_peft_model, TaskType

# # ✅ 1) Login
# login(token="hf....nN")

# # ✅ 2) Paths
# base_path = "E:/llm-fine-tuning"
# data_path = os.path.join(base_path, "data.json")
# output_dir = os.path.join(base_path, "falcon1b-lora-finetuned")
# logs_dir = os.path.join(base_path, "logs")
# hf_cache = os.path.join(base_path, "hf-cache")

# print(f"✅ Using base path: {base_path}")

# # ✅ 3) Load dataset
# with open(data_path, "r", encoding="utf-8") as f:
#     data = json.load(f)
# print(f"✅ Loaded {len(data)} examples")

# dataset = Dataset.from_list(data)

# # ✅ 4) Load Falcon-1B with safe 4-bit quantization
# base_model = "tiiuae/falcon-rw-1b"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype="float16",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
#     print("✅ pad_token was missing — set to eos_token.")

# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="auto",
#     quantization_config=bnb_config,
#     cache_dir=hf_cache
# )

# # ✅ Falcon + LoRA best practice for checkpointing
# model.config.use_cache = False
# model.config.output_attentions = False
# model.config.output_hidden_states = False
# print("✅ use_cache disabled for gradient checkpointing.")

# # ✅ 5) LoRA config
# lora_config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["query_key_value", "dense"],
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )
# model = get_peft_model(model, lora_config)

# # ✅ CRITICAL FIX: enable input embedding grads
# model.enable_input_require_grads()
# print(f"✅ LoRA injected: {lora_config.target_modules} + input grads enabled.")

# # ✅ 6) Robust tokenizer: causal LM mask
# def tokenize(example):
#     full = example["prompt"] + example["response"]
#     tokenized = tokenizer(
#         full,
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     labels = tokenized["input_ids"].copy()
#     prompt_ids = tokenizer(
#         example["prompt"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )["input_ids"]
#     prompt_len = sum(1 for i in prompt_ids if i != tokenizer.pad_token_id)
#     labels[:prompt_len] = [-100] * prompt_len  # mask prompt
#     tokenized["labels"] = labels
#     return tokenized

# print("✅ Tokenizing...")
# tokenized = dataset.map(tokenize, batched=False)
# print("✅ Tokenization done.")

# # ✅ 7) TrainingArguments: safe for 8GB
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=1,
#     num_train_epochs=3,
#     learning_rate=2e-4,
#     fp16=True,
#     gradient_checkpointing=True,
#     logging_dir=logs_dir,
#     save_strategy="epoch",
#     logging_strategy="steps",
#     logging_steps=10,
#     report_to="none"
# )

# # ✅ 8) Trainer with safe collator
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized,
#     data_collator=default_data_collator
# )

# print("✅ Training starts NOW!")
# trainer.train()

# # ✅ 9) Save final adapter + tokenizer
# model.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)
# print(f"\n🎉✅ DONE! LoRA adapter & tokenizer saved to: {output_dir}")





# finetune_falcon_lora_v2.py
import os
import json
from datasets import Dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

# =============== 1) Login to HF Hub (optional but recommended)
login(token="hf....nN")

# =============== 2) Paths (adjust if needed)
BASE_PATH = "E:/llm-fine-tuning"
DATA_PATH = os.path.join(BASE_PATH, "data.json")  # Your dataset: list of dicts with 'conversation' and 'summary'
OUTPUT_DIR = os.path.join(BASE_PATH, "falcon1b-lora-finetuned")
LOGS_DIR = os.path.join(BASE_PATH, "logs")
HF_CACHE = os.path.join(BASE_PATH, "hf-cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print(f"✅ Using base path: {BASE_PATH}")

# =============== 3) Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✅ Loaded {len(data)} examples")

dataset = Dataset.from_list(data)

# =============== 4) Add prompt and response fields for instruction tuning
def build_prompt_response(example):
    # Example is a dict with keys as columns
    # Defensive: convert to string to avoid type issues
    conversation = example.get("conversation")
    summary = example.get("summary")
    if conversation is None or summary is None:
        raise ValueError(f"Missing keys in example: {example.keys()}")

    prompt = (
        "以下の日本語の医療会話を読み、指定された形式で要約してください：\n\n"
        "会話:\n" + str(conversation) + "\n\n要約:\n"
    )
    response = str(summary)
    return {"prompt": prompt, "response": response}


dataset = dataset.map(build_prompt_response)

# =============== 5) Load Falcon-RW-1B with 4-bit quantization
MODEL_NAME = "tiiuae/falcon-rw-1b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ pad_token was missing — set to eos_token.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)

# =============== 6) Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# Disable cache for gradient checkpointing and enable input grads for LoRA
model.config.use_cache = False
model.enable_input_require_grads()

print(f"✅ LoRA applied on {lora_config.target_modules}, use_cache disabled, input grads enabled.")

# =============== 7) Tokenization & causal masking function
MAX_LENGTH = 512  # Adjust depending on GPU VRAM

def tokenize_and_mask(example):
    full_text = example["prompt"] + example["response"]

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    tokenized_prompt = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    labels = input_ids.copy()

    # Mask prompt tokens for loss calculation (-100 means ignore)
    prompt_len = sum(1 for id in tokenized_prompt["input_ids"] if id != tokenizer.pad_token_id)
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

print("✅ Tokenizing dataset with causal masking...")
tokenized_dataset = dataset.map(tokenize_and_mask, batched=False)
print("✅ Tokenization done.")

# =============== 8) Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    logging_dir=LOGS_DIR,
    save_strategy="steps",
    save_steps=100,
    logging_strategy="steps",
    logging_steps=10,
    report_to="none",
    save_total_limit=2,
)


# =============== 9) Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator,
)

print("✅ Starting training...")
trainer.train()

# =============== 10) Save model and tokenizer
print("✅ Saving model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"🎉 Fine-tuning complete! Saved to: {OUTPUT_DIR}")

"""
Teach Llama-3-8B (or Mistral) to use the <think> format.
Why Unsloth? It patches the RoPE embeddings and Triton kernels,
making this step take ~20 mins instead of 2 hour
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# CONFIG
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)

# LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loft_config = None,
)

# Data preparation (Format: "User: Q \n Assistant: <think>...")
dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    conversations = examples["conversations"]
    texts = []
    for conv in conversations:
        # conv[0] is User, conv[1] is Assistant (with <think>)
        text = f"User: {conv[0]['content']}\n\nAssistant: {conv[1]['content']}" + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)

# Train (SFT)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 300, # Quick cold start (approx 30 mins)
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "models/sft_v1",
        optim = "adamw_8bit", # 8-bit optimizer saves VRAM
        seed = 3407,
    ),
)

trainer.train()
model.save_pretrained("models/sft_v1") # Save Adapters
tokenizer.save_pretrained("models/sft_v1")
print("SFT Model Saved.")

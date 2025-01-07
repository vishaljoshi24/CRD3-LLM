from load_data import CRD3
from datasets import DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
import os

# Initialize the CRD3 dataset builder
crd3_builder = CRD3()
crd3_builder.download_and_prepare()

# Load the dataset splits
dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

# Load the tokenizer and model
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Add custom padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'
model.resize_token_embeddings(len(tokenizer))

# Pre-process the dataset
df = pd.DataFrame(dataset["train"])
def extract_turn_data(turns):
    inputs, labels = [], []
    for j in range(1, len(turns)):
        prev_turn = turns[j-1]
        current_turn = turns[j]
        context = " ".join(
            f"{name}: {utterance}"
            for name, utterance in zip(prev_turn["names"], prev_turn["utterances"])
        )
        target = " ".join(
            f"{name}: {utterance}"
            for name, utterance in zip(current_turn["names"], current_turn["utterances"])
        )
        inputs.append(context)
        labels.append(target)
    return " ".join(inputs), " ".join(labels)

df["inputs"], df["labels"] = zip(*df["turns"].apply(extract_turn_data))

# Tokenize the dataset
def tokenize_and_save(dataset, output_dir):
    tokenized_inputs = tokenizer(
        dataset["inputs"].explode().tolist(),
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    tokenized_labels = tokenizer(
        dataset["labels"].tolist(),
        max_length=64,
        truncation=True,
        padding="max_length",
    )
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    hf_dataset = Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_inputs["labels"],
    })
    hf_dataset.save_to_disk(output_dir)

# Save preprocessed dataset
data_dir = "processed_dataset"
if not os.path.exists(data_dir):
    tokenize_and_save(df, data_dir)

# Load dataset dynamically
train_dataset = load_from_disk(data_dir)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1000,  # Reduce logging frequency
    save_steps=2000,  # Save less frequently
    fp16=False,  # Disable mixed precision to avoid memory spikes
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch size
    report_to="none",
    no_cuda=False  # Enable CUDA if available
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

import torch
import os
from load_data import CRD3
from math import exp
from collections import Counter
from datasets import DatasetDict, load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd


# Initialize the CRD3 dataset builder
crd3_builder = CRD3()
crd3_builder.download_and_prepare()

# Load the dataset splits
dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

# Pre-process the dataset
df = pd.DataFrame(dataset["test"])
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
data_path = "processed_test_dataset"
if not os.path.exists(data_path):
    tokenize_and_save(df, data_path)

model_path = "./results"
 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

test_dataset = load_from_disk(data_path)
model.resize_token_embeddings(len(tokenizer))

# Function to compute perplexity
def compute_perplexity(model, tokenizer, dataset):
    total_loss = 0
    num_batches = 0
    
    for sample in dataset:
        inputs = tokenizer(sample["inputs"], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return exp(avg_loss)

# Function to compute Dist-n metrics
def compute_distinct_n(texts, n):
    ngrams = [tuple(text.split()[i:i+n]) for text in texts for i in range(len(text.split()) - n + 1)]
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams) if ngrams else 0

def generate_samples(model, tokenizer, prompts, num_samples=100, max_length=50):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return [generator(prompt, max_length=max_length)[0]["generated_text"] for prompt in prompts[:num_samples]]

# Function to compute recall at k
def recall_at_k(model, tokenizer, dataset, k=5):
    correct_count = 0
    total_count = 0
    
    for sample in dataset:
        inputs = tokenizer(sample["inputs"], return_tensors="pt", padding=True, truncation=True)
        labels = tokenizer(sample["labels"], return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -labels.size(-1):, :]  # Focus on last tokens
        
        top_k = torch.topk(logits, k=k, dim=-1).indices
        correct = (top_k == labels.unsqueeze(-1)).sum().item()
        
        correct_count += correct
        total_count += labels.numel()

    return correct_count / total_count

# Evaluation Execution

# Perplexity
ppl = compute_perplexity(model, tokenizer, test_dataset)
print(f"Perplexity: {ppl}")

# Diversity Metrics (Dist-1, Dist-2, Dist-3)
sample_prompts = ["Sample prompt 1", "Sample prompt 2"]  # Replace with relevant prompts
generated_texts = generate_samples(model, tokenizer, sample_prompts)

for n in range(1, 4):
    dist_n = compute_distinct_n(generated_texts, n)
    print(f"Dist-{n}: {dist_n}")

# Recall at 5
recall_5 = recall_at_k(model, tokenizer, test_dataset, k=5)
print(f"Recall at 5: {recall_5}")

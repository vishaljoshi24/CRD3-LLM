from load_data import CRD3
from datasets import DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

# Initialize CRD3 and load dataset
crd3_builder = CRD3()
crd3_builder.download_and_prepare()

dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

# Initialize tokenizer and model
checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Define padding token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Pre-process the dataset
train_df = pd.DataFrame(dataset["train"])
#val_df = pd.DataFrame(dataset["validation"])

# Specify the list of actors you want to filter by
actors_of_interest = ["LAURA"]  # Replace with actual actor names

def extract_turn_data(turns):
    inputs, labels = [], []
    for j in range(1, len(turns)):
        # Include all previous turns as context
        prev_turns = [
            f"{utterance}" 
            for i in range(j) if "utterances" in turns[i] 
            for utterance in zip(turns[i]["utterances"])
        ]
        
        # Only extract LAURA's responses
        current_turn = turns[j]
        laura_responses = [
            utterance for name, utterance in zip(current_turn["names"], current_turn["utterances"])
            if name in actors_of_interest
        ]

        context = " ".join(prev_turns) if prev_turns else ""
        target = " ".join(laura_responses) if laura_responses else ""

        if context and target:
            inputs.append(context)
            labels.append(target)

    return inputs, labels

train_df["turns"].drop_duplicates()
#val_df["turns"].drop_duplicates()
train_df["inputs"], train_df["labels"] = zip(*train_df["turns"].apply(extract_turn_data))
#val_df["inputs"], val_df["labels"] = zip(*val_df["turns"].apply(extract_turn_data))


def tokenize_and_save(dataset, output_dir):
    tokenized_inputs = tokenizer(
        dataset["inputs"].tolist(),
        max_length=512,
        truncation=True,
        padding="max_length",
    )
    tokenized_labels = tokenizer(
        dataset["labels"].tolist(),
        max_length=512,
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

# Save preprocessed datasets
train_data_dir = "new_laura_dataset"
#val_data_dir = "new_laura_val_dataset"

if not os.path.exists(train_data_dir):
    tokenize_and_save(train_df, train_data_dir)
# if not os.path.exists(val_data_dir):  
#     tokenize_and_save(val_df, val_data_dir)
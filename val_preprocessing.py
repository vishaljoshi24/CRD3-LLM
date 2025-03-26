from load_data import CRD3
from datasets import DatasetDict, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

crd3_builder = CRD3()
crd3_builder.download_and_prepare()

dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Pre-process the dataset
val_df = pd.DataFrame(dataset["validation"])
def extract_turn_data(turns):
    inputs, labels = [], []
    for j in range(1, len(turns)):
        prev_turn = turns[j-1] if j-1 >= 0 else {"utterances": [""]}
        current_turn = turns[j] if j < len(turns) else {"utterances": [""]}
        context = " ".join(
            f"{utterance}"
            for utterance in prev_turn["utterances"]
        ) if prev_turn["utterances"] else ""

        target = " ".join(
            f"{utterance}"
            for utterance in current_turn["utterances"]
        ) if current_turn["utterances"] else ""
        inputs.append(context)
        labels.append(target)
    
    return " ".join(inputs), " ".join(labels)

val_df["turns"].drop_duplicates()
val_df["inputs"], val_df["labels"] = zip(*val_df["turns"].apply(extract_turn_data))

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

# Save preprocessed dataset
data_dir = "validation_dataset"
if not os.path.exists(data_dir):
    tokenize_and_save(val_df, data_dir)

validation_dataset = load_from_disk(data_dir)

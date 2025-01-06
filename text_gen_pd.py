from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer
import evaluate
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from datasets import Dataset

# Initialize the CRD3 dataset builder
crd3_builder = CRD3()

# Ensure the dataset is downloaded and prepared
crd3_builder.download_and_prepare()

# Load the dataset splits
dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

#print(dataset["train"][0])

checkpoint = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Add a custom pad token (e.g., '[PAD]')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Now set the pad_token explicitly
tokenizer.pad_token = '[PAD]'

prefix = "Given the history and current utterance predict next utterance: "

#print(dataset["train"][1]["turns"])
print(len(dataset["train"][2]["turns"]))

df = pd.DataFrame(dataset["train"]) #Convert training set into a pandas DataFrame

def extract_turn_data(turns):
    inputs, labels = [], []
    for j in range(1, len(turns)):
        prev_turn = turns[j-1]
        current_turn = turns[j]

        # Construct context and target strings
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

    # Return flattened lists for easier tokenization
    return " ".join(inputs), " ".join(labels)

df["inputs"], df["labels"] = zip(*df["turns"].apply(extract_turn_data))

tokenized_dataset = tokenizer(
    df["inputs"].explode().tolist(),
    max_length=1024,
    truncation=True,
    padding="max_length",
)

labels_tokenized = tokenizer(
    df["labels"].tolist(),  # Convert to List[str]
    max_length=128,
    truncation=True,
    padding="max_length",
)

# Add labels to the tokenized dataset
tokenized_dataset["labels"] = labels_tokenized["input_ids"]
model.resize_token_embeddings(len(tokenizer))

def prepare_for_training(tokenized_dataset, labels):
    return Dataset.from_dict({
        "input_ids": tokenized_dataset["input_ids"],
        "attention_mask": tokenized_dataset["attention_mask"],
        "labels": labels,  # Ensure this matches the "labels" structure
    })

train_dataset = prepare_for_training(tokenized_dataset, labels_tokenized["input_ids"])

training_args = TrainingArguments(
    output_dir="./results",               # Directory to save checkpoints and outputs
    evaluation_strategy="no",         # Evaluate after every epoch
    learning_rate=5e-5,                   # Learning rate
    per_device_train_batch_size=8,       # Batch size per device for training
    per_device_eval_batch_size=8,        # Batch size per device for evaluation
    num_train_epochs=3,                  # Number of epochs
    weight_decay=0.01,                   # Weight decay for regularization
    save_total_limit=2,                  # Save only the last 2 checkpoints
    logging_dir="./logs",                # Directory for logs
    logging_steps=500,                   # Log every 500 steps
    save_steps=1000,                     # Save model every 1000 steps
    fp16=True,                           # Enable mixed precision training (if using a GPU with FP16 support)
    report_to="none",                    # Avoid reporting to external tools like WandB
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
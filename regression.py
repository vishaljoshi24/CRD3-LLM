import numpy as np
from sklearn.metrics import mean_squared_error
import evaluate
from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

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

# Load tokenizer for BERT model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# Preprocess the dataset to tokenize inputs and add labels
def preprocess_function(examples):
    tokenized = tokenizer(examples["chunk"], padding="max_length", truncation=True)
    tokenized["labels"] = examples["alignment_score"]  # Add alignment_score as labels for regression
    return tokenized

# Tokenize the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split the tokenized datasets for training and evaluation
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# Load model for regression
model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased",
    num_labels=1,  # For regression tasks
    torch_dtype="auto"
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    eval_strategy="epoch",
    no_cuda=True  # Set to False if you want to use GPU
)

# Define compute_metrics for regression (e.g., mean squared error)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze(-1)  # Flatten logits to match labels
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

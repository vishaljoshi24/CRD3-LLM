from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForNextSentencePrediction
from transformers import TrainingArguments

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

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer(examples["chunk"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForNextSentencePrediction.from_pretrained("google-bert/bert-base", num_labels=7, torch_dtype="auto")

training_args = TrainingArguments(output_dir="test_trainer")

from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer

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

class ChunkAndTurnTokenizer(AutoTokenizer):


    def tokenize_function(examples):
        
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        tokenized_data = {}

        # Tokenize the "chunk" column
        if "chunk" in examples:
            tokenized_data["chunk"] = tokenizer(
                examples["chunk"], padding="max_length", truncation=True
            )["input_ids"]

        # Tokenize the "turns" column
        if "turns" in examples:
            tokenized_utterances = []
            tokenized_names = []

            for turns in examples["turns"]:
                batch_utterances = []
                batch_names = []
                for turn in turns:
                    # Tokenize utterances and names for this turn
                    utterances = turn.get("utterances", [])
                    batch_utterances.append(
                        tokenizer(utterances, padding="max_length", truncation=True)["input_ids"]
                    )
                    names = turn.get("names", [])
                    batch_names.append(
                        tokenizer(names, padding="max_length", truncation=True)["input_ids"]
                    )

                # Append tokenized batches for this row
                tokenized_utterances.append(batch_utterances)
                tokenized_names.append(batch_names)

            # Assign to tokenized_data
            tokenized_data["utterances"] = tokenized_utterances
            tokenized_data["names"] = tokenized_names

        return tokenized_data


    tokenized_datasets = dataset.map(tokenize_function, batched=True)
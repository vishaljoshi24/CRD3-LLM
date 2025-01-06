from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer
import evaluate
import numpy as np
from transformers import AutoModelForCausalLM

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

# Add a custom pad token (e.g., '[PAD]')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Now set the pad_token explicitly
tokenizer.pad_token = '[PAD]'

prefix = "Given the history and current utterance predict next utterance: "

# print(dataset["train"][1]["turns"][2]["names"])
print(len(dataset["train"][2]["turns"]))

def preprocess_function(examples):
    # speaker = []
    # dialogue = []
    turns = []
    inputs = []  # Will hold the previous turns as context
    labels = []  # Will hold the next turn (target) for each input
    
    # Iterate over the turns starting from the 1st turn (j=0)
    for j in range(1, len(examples["turns"])):
        # Get the current turn's speaker and dialogue
        # prev_names = examples["turns"][j-1]["names"]
        # prev_utterances = examples["turns"][j-1]["utterances"]
        prev_turns = examples["turns"][j-1]
        # Add the speaker and dialogue to the context
        # speaker.append(prev_names)
        # dialogue.append(prev_utterances)
        turns.append(prev_turns)
        # The context will be the concatenation of all prior turns up to the current one
        # context = " ".join(f"{s}: {d}" for s, d in zip(speaker, dialogue))
        context = " ".join(f"{t}" for t in zip(turns))
        
        # The target is the current turn (the one we're trying to predict)
        # current_names = examples["turns"][j]["names"]
        # current_utterances = examples["turns"][j]["utterances"]
        current_turn = examples["turns"][j]
        # target = f"{current_names}: {current_utterances}"
        target = f"{current_turn}"

        # Add the context (input) and target (label) to the lists
        inputs.append(context)
        labels.append(target)

    # Tokenize the inputs and labels
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    labels_tokenized = tokenizer(labels, max_length=128, truncation=True, padding="max_length")

    # Add labels to the tokenized inputs
    model_inputs["labels"] = labels_tokenized["input_ids"]

    return model_inputs


sample = dataset["train"][1:7]

#tokenized_dataset = dataset.map(preprocess_function, batched=True)

#print(preprocess_function(sample))
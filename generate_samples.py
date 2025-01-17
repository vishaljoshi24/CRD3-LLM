from load_data import CRD3
from math import exp
from collections import Counter
from datasets import DatasetDict
import pandas as pd
from nltk import ngrams


# Initialize the CRD3 dataset builder
crd3_builder = CRD3()
crd3_builder.download_and_prepare()

# Load the dataset splits
dataset = DatasetDict({
    "train": crd3_builder.as_dataset(split="train"),
    "test": crd3_builder.as_dataset(split="test"),
    "validation": crd3_builder.as_dataset(split="validation"),
})

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
        
vals = [y for x in df['inputs'] for y in x.split()]

n = [3]
a = pd.Series([y for x in n for y in ngrams(vals, x)]).value_counts()
print (a)
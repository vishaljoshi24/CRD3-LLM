from load_data import CRD3
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

# Convert test dataset to a DataFrame
df = pd.DataFrame(dataset["test"])
print(df["turns"])

def extract_all_words(turns):
    """Extract all words from the dataset for n-gram calculation."""
    all_words = []
    for turn_list in turns:  # Iterate through each row of the "turns" column
        if isinstance(turn_list, list):  # Ensure it's a list of dictionaries
            for turn in turn_list:
                if isinstance(turn, dict) and "utterances" in turn:  # Validate structure
                    for utterance in turn["utterances"]:
                        all_words.extend(utterance.split())  # Split utterances into words
    return all_words

# Extract all words from the "turns" column
all_words = extract_all_words(df["turns"])

# Calculate n-grams in the range 1-4
all_ngrams = []
for n in range(1, 5):
    ngram_counts = Counter(ngrams(all_words, n))
    all_ngrams.extend([" ".join(ngram) for ngram in ngram_counts.keys()])

# Randomly select 100 n-grams
if len(all_ngrams) > 0:
    ngram_sample = pd.Series(all_ngrams).sample(n=min(100, len(all_ngrams)), random_state=42).tolist()
    print("Generated 100 n-grams:", ngram_sample)
else:
    print("No n-grams found.")

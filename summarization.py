from load_data import CRD3
from datasets import DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

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

print(dataset["train"][0])

checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "Summarize: "

def preprocess_function(examples):
    # Extract and concatenate all the utterances into one string for each instance
    inputs = []
    
    # Iterate over each turn in "turns" (which is a list of dictionaries)
    for turn in examples["turns"]:
        # Concatenate the utterances for the current turn
        utterance_str = " ".join(turn[0])  # Access the "utterances" in the current turn
        inputs.append(prefix + utterance_str)  # Add the prefix to the concatenated utterances

    # Tokenize the inputs
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    # Prepare labels (i.e., summarize the chunk text)
    labels = tokenizer(text_target=examples["chunk"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in  result.items()} 

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="CRD3 Summariser",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True, #change to bf16=True for XPU
    push_to_hub=False,
    no_cuda=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
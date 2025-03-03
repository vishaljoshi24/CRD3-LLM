from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from accelerate import Accelerator

# Initialize the Accelerator for distributed training
accelerator = Accelerator()

# Load your preprocessed dataset directly from disk
data_dir = "processed_dataset"  # Change this to your actual dataset directory
train_dataset = load_from_disk(data_dir)

# Load your model and tokenizer from HuggingFace Hub
model_name = "vishaljoshi24/crd3_text_gen"  # Replace with the name of your model repo on HuggingFace
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add custom padding token (if necessary)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = '[PAD]'
model.resize_token_embeddings(len(tokenizer))

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results-1",  # This will save the model in the HuggingFace model repo
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size if necessary
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=1000,  # Reducing logging frequency
    save_steps=2000,  # Saving less frequently
    fp16=False,  # Disabling mixed precision to avoid memory spikes
    gradient_accumulation_steps=8,  # Accumulating gradients to simulate larger batch size
    report_to="none",
    push_to_hub=True,  # Automatically push the model to HuggingFace Hub after training
    hub_strategy="end",  # Push to hub after training finishes
    no_cuda=False
)

# Initialize the Trainer with the Accelerator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Prepare the model and trainer for distributed training
model, trainer = accelerator.prepare(model, trainer)

# Train the model
trainer.train()

# Push the updated model and tokenizer to HuggingFace Hub
model.push_to_hub(model_name)  # Ensure this matches your HuggingFace repo name
tokenizer.push_to_hub(model_name)  # Ensure this matches your HuggingFace repo name

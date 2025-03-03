from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from rp_proc import train_dataset, tokenizer, model

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",
    evaluation_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduce batch size
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
    no_cuda=True  
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

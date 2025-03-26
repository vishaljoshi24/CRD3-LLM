import optuna
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_from_disk
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load datasets
train_dataset = load_from_disk('laura_dataset')
validation_dataset = load_from_disk('laura_val_dataset')

# Reduce training dataset size to match validation dataset size
train_dataset = train_dataset.select(range(len(validation_dataset)))

def objective(trial):
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2])

    os.makedirs('output/hyperparameters', exist_ok=True)

    train_args = TrainingArguments(
        output_dir='output/hyperparameters',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        fp16=True,
        logging_dir='output/logs',
        logging_steps=10,
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )

    logger.info(f"Starting trial with lr={learning_rate}, batch_size={batch_size}")
    trainer.train()
    eval_result = trainer.evaluate()
    
    logger.info(f"Trial completed with eval_loss={eval_result['eval_loss']}")
    return eval_result['eval_loss']

def run_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)
    
    best_params_path = "output/best_hyperparameters.txt"
    with open(best_params_path, "w") as f:
        f.write(str(study.best_params))
    
    logger.info(f"Best hyperparameters: {study.best_params}")
    print("Best hyperparameters:", study.best_params)

if __name__ == '__main__':
    try:
        logger.info("Starting hyperparameter tuning...")
        run_optimization()
    except Exception as e:
        logger.error(f"Error occurred: {e}")

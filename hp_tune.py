import optuna
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from rp_proc import train_dataset
from val_proc import validation_dataset

def objective(trial):
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

    learning_rate = trial.suggest_loguniform('learing_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])

    train_args = TrainingArguments(
        output_dir='./hyperparameters',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset
    )

    trainer.train()
    eval_result = trainer.evaluate()

    return eval_result['eval_loss']

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

print("Best hyperparameters:", study.best_params)
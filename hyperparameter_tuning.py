import subprocess
import optuna

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    num_epochs = trial.suggest_int("num_epochs", 3, 10)
    weight_decay = trial.suggest_uniform("weight_decay", 0.0, 0.1)

    # Call the training script with the suggested parameters
    result = subprocess.run(
        [
            "python", "text_gen_tuning.py",
            "--learning_rate", str(learning_rate),
            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--weight_decay", str(weight_decay),
        ],
        capture_output=True,
        text=True
    )

    # Extract loss or another metric from the output
    for line in result.stdout.splitlines():
        if "eval_loss" in line:
            return float(line.split(":")[-1])

# Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)

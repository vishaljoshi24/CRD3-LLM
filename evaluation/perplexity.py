import torch
from math import exp
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_perplexity(model, dataset, batch_size=8):
    total_loss = 0
    total_tokens = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        input_ids = torch.tensor([sample["input_ids"] for sample in batch]).to(device)
        attention_mask = torch.tensor([sample["attention_mask"] for sample in batch]).to(device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)
        total_loss += outputs.loss.item() * input_ids.size(1)  # Scale by sequence length
        total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    return exp(avg_loss)

if __name__ == "__main__":
    model_path = "./results/checkpoint-7305"
    data_path = "processed_test_dataset"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    test_dataset = load_from_disk(data_path)
    test_dataset = list(test_dataset)  # Convert to list for batching

    perplexity = compute_perplexity(model, test_dataset)
    print(f"Perplexity: {perplexity}")

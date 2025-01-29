import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

def recall_at_k(model, dataset, k=5, batch_size=8):
    correct_count = 0
    total_count = 0

    device = "cpu"
    model = model.to(device)

    dataset = list(dataset)  # Convert to list for easier batching

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        input_ids = torch.tensor([sample["input_ids"] for sample in batch]).to(device)
        attention_mask = torch.tensor([sample["attention_mask"] for sample in batch]).to(device)
        labels = torch.tensor([sample["labels"] for sample in batch]).to(device)

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -labels.size(1):, :]  # Focus on last tokens

        # Compute top-k predictions
        top_k = torch.topk(logits, k=k, dim=-1).indices
        correct = (top_k == labels.unsqueeze(-1)).sum().item()

        correct_count += correct
        total_count += labels.numel()

    return correct_count / total_count

if __name__ == "__main__":
    model_path = "openai-community/gpt2"
    data_path = "processed_test_dataset"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    test_dataset = load_from_disk(data_path)
    test_dataset = list(test_dataset)  # Convert to list for batching

    recall_5 = recall_at_k(model, test_dataset, k=5)
    print(f"Recall at 5: {recall_5}")

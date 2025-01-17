from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def compute_distinct_n(texts, n):
    ngrams = [tuple(text.split()[i:i+n]) for text in texts for i in range(len(text.split()) - n + 1)]
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams) if ngrams else 0

def generate_samples(model, tokenizer, prompts, num_samples=100, max_length=50):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return [generator(prompt, max_length=max_length)[0]["generated_text"] for prompt in prompts[:num_samples]]

if __name__ == "__main__":
    model_path = "./results/checkpoint-7305"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    sample_prompts = ["Make a perception check", "I pick up", "I'm going to"]
    generated_texts = generate_samples(model, tokenizer, sample_prompts)

    for n in range(1, 4):
        dist_n = compute_distinct_n(generated_texts, n)
        print(f"Dist-{n}: {dist_n}")

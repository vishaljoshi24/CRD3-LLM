from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import numpy as np
import os

random.seed(42)
np.random.seed(42)

def generate_samples(model, tokenizer, prompts, num_samples=100, max_length=100):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.7, top_p=0.9, top_k=5)
    return [generator(prompt, max_length=max_length)[0]["generated_text"] for prompt in prompts[:num_samples]]

if __name__ == "__main__":
    HF_TOKEN = os.getenv("HF_TOKEN")
    model_path = "vishaljoshi24/crd3_text_gen_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    sample_prompts = ["""Where do you hail from?"""]
    generated_texts = generate_samples(model, tokenizer, sample_prompts)
    print(generated_texts)
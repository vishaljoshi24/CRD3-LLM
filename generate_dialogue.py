from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import os

#random.seed(42)
np.random.seed(42)

def generate_samples(model, tokenizer, prompts, num_samples=100, max_length=200, system_prompt=None):
    responses = []
    for p in prompts[:num_samples]:
        full_prompt = f"{system_prompt}\n\n{p}" if system_prompt else p
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.cuda()
        attention_mask = inputs.attention_mask.cuda()
        
        #print("Input Text:", full_prompt)
        #print("Tokenized Input:", tokenizer.convert_ids_to_tokens(input_ids[0]))
        #print("Input Shape:", input_ids.shape)
        
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_length,
            temperature=0.7,  # Increase randomness
            top_k=50,         # Use top-k sampling
            top_p=0.9,       # Use nucleus sampling
            do_sample=True,   # Enable sampling
            repetition_penalty=1.1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        #print("Generated IDs:", output_ids.tolist())

        generated_tokens = output_ids[0][input_ids.shape[1]:]
        #print("Generated Tokens:", generated_tokens.tolist())
        response = tokenizer.decode(generated_tokens, skip_special_tokens=False)  # Show all tokens initially
        #print("Decoded Response:", response)
        responses.append(response.strip())
    return responses

if __name__ == "__main__":
    HF_TOKEN = os.getenv("HF_TOKEN")
    model_path = "vishaljoshi24/crd3_dialogue"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|startoftext|>"
    tokenizer.eos_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(model_path).cuda()
    model.eval()

    system_prompt = (
        """You are a player in a Dungeons & Dragons game. Stay in character, respond as if you are part of the game. "
            Don't act as the Dungeon Master. Play along, describe your actions, interact with others naturally."""
    )

    sample_prompts = [
        """We have been hiding out in Ank'Harel for several weeks now. We don't know that many people here but
        we have been getting to know the those few regulars at the inn that aren't too drunk to hold a conversation.
        That one man with the dragon ancestry, you thought he was stupid and weird. Why about him made you think that?"""
    ]

    generated_texts = generate_samples(model, tokenizer, sample_prompts, system_prompt=system_prompt)
    print(generated_texts)
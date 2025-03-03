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

    sample_prompts = ['Step...', 'sigh) And', 'closer, and there', "That's actually,", 'Yes, the Pablove', 'out. Oh my', 'a rib, and', 'Making barrels.', '15 for one. 15', 
                      'I may, Orly,', 'Yes. Perhaps', "well, that's breaking the", 'Yeah, 15 feet around', 'become,', 'contains the Divine', 'moves-- To', 'wolfstarumbra.', 
                      'next time to', 'foot in', 'casting this', 'next round of combat.', 'friends. Come." She jumps', 'Caleb;', 'reaction, then, to', 'a nap with one', 
                      'you have to compensate', 'And when', "drink and he's", "I'm wet.", 'important to not hesitate', 'hear it and the', 
                      'you? She goes,', 'to go talk to', 'watch as Caduceus comes', 'paper. Just a', 'saw him look', "just-- I mean, I'm", 
                      'you catch in the', 'Eventually making', 'vote for sailing up', 'smiling and finding', 'other shipmates...', 'five copper.',
                       'Not in your readings,', 'hiding. Want', 'dust. I', "I walk out. She'll", 'the crew. What', 'brightly, her', 
                       'up? No, the', 'a tattoo.', 'to do! You', 'like football,', 'enter with', 'walk around the', 'in the background as', 'loot and', 'when you saw us', 
                      'and slammed', 'you, Laura, and', 'but of', 'here at Critical Role', 'explosion on the left', 
                      'this playful', 'I create', 'It catches your', 'a faint hint of', 'much resistance', 'Got to be careful', 
                      'You roll an acrobatics', 'him! He', 'roof. He looks back', 
                      'fun, though. Listen, I', 'so I will', "to-- It's on", "than Yasha's", 'that prevents', 'essentially give the',
                        "the boat-- Right? It's", 'their second', 'day. Okay, Fjord.', "don't know a", 'action seeing', 'navigator." Very cool.', 'male,', 'exhaling,', 'of correspondence. Well as', 
                      'the door here bursts', 'of a sudden? Or', 'shirt off!', 'Maybe she just likes', 'expensive makeup.', 'of thick material',
                        'random.', 'is my home."', 'Never leave again! Okay,', 'it down to nine.', "minutes. You didn't", 'have. Have', 'help and sense']
    generated_texts = generate_samples(model, tokenizer, sample_prompts)

    for n in range(1, 4):
        dist_n = compute_distinct_n(generated_texts, n)
        print(f"Dist-{n}: {dist_n}")

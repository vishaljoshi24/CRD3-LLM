from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your already trained model and tokenizer (assuming they are saved locally)
model = AutoModelForCausalLM.from_pretrained("./results/checkpoint-7305")
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-7305")

# Specify the name of your HuggingFace model repository
model_name = "vishaljoshi24/crd3_dialogue_generator"  # Replace with your model repo name on HuggingFace

# Push the model and tokenizer to the HuggingFace Hub
model.push_to_hub(model_name)
tokenizer.push_to_hub(model_name)

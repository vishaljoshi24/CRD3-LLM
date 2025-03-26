import os
import gradio as gr

api_key = os.getenv("HF_TOKEN")

def start_training():
    os.system("python utterance_train.py --model_name_or_path openai-community/gpt2 --data_dir tokenized_utterances --api_key HF_TOKEN" )
    return "Evaluation completed!"

iface = gr.Interface(
    fn=start_training,
    inputs=[],
    outputs="text",
    live=True,
)

iface.launch()

from llama_index.core import SimpleDirectoryReader
from utils import setup_index_and_chat_engine, get_embedding_model, set_llm, set_chat_memory
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch
import os
import glob

DIRECTORY_PATH = "data"
EMBED_MODEL = get_embedding_model()

def load_docs():
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    for file_path in all_files:
        reader = SimpleDirectoryReader(input_files=[file_path]).load_data()
        documents.extend(reader)
    return documents


def create_chat_engine(model_provider, model, temperature, max_tokens, custom_prompt, top_p):
    documents = load_docs()
    embed_model = EMBED_MODEL

    if model_provider == "Ollama":
        llm = set_llm(model, temperature, max_tokens)
    elif model_provider == "Hugging Face":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        llm = HuggingFaceLLM(
            model_name=model,
            tokenizer_name=model,
            context_window=2048,  # Adjust as needed
            max_new_tokens=max_tokens,
            model_kwargs={"quantization_config": quantization_config},
            generate_kwargs={
                "temperature": temperature,
                "top_p": top_p,
            },
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")

    memory = set_chat_memory(model)
    return setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory,
                                       custom_prompt=custom_prompt)

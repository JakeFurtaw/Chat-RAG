from llama_index.core import SimpleDirectoryReader
from utils import (setup_index_and_chat_engine, get_embedding_model, set_chat_memory,
                   set_ollama_llm, set_huggingface_llm, set_nvidia_model)
import torch, os, glob, gc

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


def create_chat_engine(model_provider, model, temperature, max_tokens, custom_prompt, top_p,
                       context_window, quantization):
    # Emptying GPU Memory
    torch.cuda.empty_cache()
    gc.collect()
    #Loading Docs and Embedding Model
    documents = load_docs()
    embed_model = EMBED_MODEL
    # Choosing LLM
    if model_provider == "Ollama":
        llm = set_ollama_llm(model, temperature, max_tokens)
    elif model_provider == "HuggingFace":
        llm = set_huggingface_llm(model, temperature, max_tokens, top_p, context_window, quantization)
    elif model_provider == "NVIDIA NIM":
        llm = set_nvidia_model(model, temperature, max_tokens, top_p)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    memory = set_chat_memory(model)
    return setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory,
                                       custom_prompt=custom_prompt)

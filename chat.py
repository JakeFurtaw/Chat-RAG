from llama_index.core import SimpleDirectoryReader
from utils import setup_index_and_chat_engine, set_embedding_model, set_llm, set_chat_memory
import os
import glob

DIRECTORY_PATH = "data"


def load_docs():
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    for file_path in all_files:
        reader = SimpleDirectoryReader(input_files=[file_path]).load_data()
        documents.extend(reader)
    return documents


def create_chat_engine(model):
    documents = load_docs()
    embed_model = set_embedding_model()
    llm = set_llm(model)
    memory = set_chat_memory(model)
    return setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory)

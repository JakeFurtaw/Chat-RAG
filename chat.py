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


def main(model, send_input=input, input_print=input) -> None:
    embed_model = set_embedding_model()
    llm = set_llm(model)
    documents = load_docs()
    memory = set_chat_memory(model)
    chat_engine = setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory)

    while True:
        query = send_input()
        if query is None:
            break
        response = chat_engine.chat(query)
        input_print(str(response))


if __name__ == "__main__":
    main("codestral:latest")

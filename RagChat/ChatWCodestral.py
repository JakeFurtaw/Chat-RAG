from llama_index.core import SimpleDirectoryReader
from utils import setup_index_and_chat_engine, load_models
import os
import glob

DIRECTORY_PATH = "data"


def load_docs():
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]  # Filter out directories

    if len(all_files) > 0:
        documents = []
        for file_path in all_files:
            reader = SimpleDirectoryReader(input_files=[file_path]).load_data()
            documents.extend(reader)
    else:
        documents = []

    return documents


def main(send_input=input, input_print=input) -> None:
    embed_model, llm = load_models()
    documents = load_docs()
    chat_engine = setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model)

    while True:
        query = send_input()
        if query.lower() == 'e':
            break
        response = chat_engine.chat(query)
        input_print(str(response))


if __name__ == "__main__":
    main()

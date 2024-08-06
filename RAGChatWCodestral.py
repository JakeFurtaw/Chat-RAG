from llama_index.core import SimpleDirectoryReader
from utils import setup_index_and_chat_engine, load_models
import os

DIRECTORY_PATH = "/data"


def has_multiple_files(directory):
    file_count = sum(1 for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item)))
    return file_count > 1


def load_docs():
    directory_path = DIRECTORY_PATH
    if has_multiple_files(directory_path):
        reader = SimpleDirectoryReader(input_dir=directory_path, recursive=True).load_data()
        documents = []
        for docs in reader:
            for doc in docs:
                documents.append(doc)
    else:
        documents = SimpleDirectoryReader(input_dir=directory_path).load_data()
    return documents


def main() -> None:
    embed_model, llm = load_models()
    documents = load_docs()
    # Setting up chat engine
    chat_engine = setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model)
    # Looping chat until user is done chatting
    while (query := input("Enter your coding question(e to exit): ")) != "e":
        response = chat_engine.chat(query)
        print(str(response))


if __name__ == "__main__":
    main()

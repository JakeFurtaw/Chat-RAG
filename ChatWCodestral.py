from llama_index.core import SimpleDirectoryReader
from utils import setup_index_and_chat_engine, load_models
import os

DIRECTORY_PATH = "data/a9aabfd1c4cad78e28fdc7bbe937b5536d48db9f"


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

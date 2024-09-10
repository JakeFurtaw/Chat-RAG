from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_parse import LlamaParse
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from utils import (setup_index_and_chat_engine, set_embedding_model, set_chat_memory,
                   set_ollama_llm, set_huggingface_llm, set_nvidia_model, set_openai_model, set_anth_model)
import torch, os, glob, gc, dotenv, chromadb
dotenv.load_dotenv()

DIRECTORY_PATH = "data"
Neo4j_DB_PATH = "Databases/Neo4j"
Chroma_DB_PATH = "Databases/ChromaDB"
Milvus_DB_PATH = "Databases/MilvusDB"
EMBED_MODEL = set_embedding_model()

# TODO Add free parsing options for advanced docs, Llama Parse only lets you parse 1000 free docs a day
# TODO Figure out why multiprocessing of docs causes program to reload in a loop
# Local Document Loading Function
def load_local_docs():
    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    supported_extensions = [".pdf", ".docx", ".xlsx", ".csv", ".xml", ".html"]
    for file in all_files:
        file_extension = os.path.splitext(file)[1].lower()
        if "LLAMA_CLOUD_API_KEY" in os.environ and file_extension in supported_extensions:
            file_extractor = {file_extension: parser}
            documents.extend(
                SimpleDirectoryReader(input_files=[file], file_extractor=file_extractor).load_data())
        else:
            documents.extend(SimpleDirectoryReader(input_files=[file]).load_data())
    return documents

# GitHub Repo Reader setup function. Sets all initial parameters and handles data load of the repository
def load_github_repo(owner, repo, branch):
    if "GITHUB_PAT" in os.environ:
        github_client = GithubClient(github_token=os.getenv("GITHUB_PAT"), verbose=True)
        owner=owner
        repo=repo
        branch=branch
        documents= GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=([".png", ".jpg", ".jpeg", ".gif", ".svg"],
                                    GithubRepositoryReader.FilterType.EXCLUDE)
        ).load_data(branch=branch)
        return documents
    else:
        print("Couldn't find your GitHub Personal Access Token in the environment file. Make sure you enter your "
              "GitHub Personal Access Token in the .env file.")


# TODO Finish and Test Vector Store implementation
# Setting up different vector stores
def setup_vector_store(vector_store, username, password, url, collection_name):
    if vector_store == "Neo4j":
        username = username
        password = password
        url = url
        embed_dim = 1536
        neo4j_vector_store = Neo4jVectorStore(username,
                                              password,
                                              url,
                                              embed_dim,
                                              database=collection_name,
                                              hybrid_search=True,
                                              distance_strategy="euclidean")
        storage_context = StorageContext.from_defaults(vector_store=neo4j_vector_store)
        return storage_context
    elif vector_store == "ChromaDB":
        chroma_client = chromadb.EphemeralClient()
        # Check to see if collection exists already
        chroma_collection = ""
        for c in chroma_client.list_collections():
            if c == collection_name:
                chroma_collection = chroma_client.get_collection(collection_name)
            else:
                chroma_collection = chroma_client.create_collection(collection_name)
        chroma_vector_store = ChromaVectorStore(chroma_collection,
                                                persist_dir=Chroma_DB_PATH)
        storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)
        return storage_context
    elif vector_store == "Milvus":
        milvus_vector_store = MilvusVectorStore(collection_name=collection_name,
                                                dim=1536,
                                                overwrite=False)
        storage_context = StorageContext.from_defaults(vector_store=milvus_vector_store)
        return storage_context
    else:
        storage_context = None
        return storage_context

# Calls setup chat engine function with model and data personalization's user inputs from front end
def create_chat_engine(model_provider, model, temperature, max_tokens, custom_prompt, top_p,
                       context_window, quantization, owner, repo, branch, vector_store, username, password, url,
                       collection_name):
    # Clearing GPU Memory
    torch.cuda.empty_cache()
    gc.collect()
    # Loading local Documents and GitHub Repos if applicable
    documents = load_local_docs()
    if owner and repo and branch:
        documents.extend(load_github_repo(owner, repo, branch))
    # Loading Storage Context if any is set by a vector store
    if vector_store is not None or "":
        storage_context = setup_vector_store(vector_store, username, password, url, collection_name)
    else:
        storage_context = None
    # Loading Embedding Model from global parameter
    embed_model = EMBED_MODEL
    # Loading LLM based off users input
    llm_setters = {
        "Ollama": lambda: set_ollama_llm(model, temperature, max_tokens),
        "HuggingFace": lambda: set_huggingface_llm(model, temperature, max_tokens, top_p, context_window, quantization),
        "NVIDIA NIM": lambda: set_nvidia_model(model, temperature, max_tokens, top_p),
        "OpenAI": lambda: set_openai_model(model, temperature, max_tokens, top_p),
        "Anthropic": lambda: set_anth_model(model, temperature, max_tokens)
    }
    try:
        llm = llm_setters[model_provider]()
    except KeyError:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    # Setting model memory
    memory = set_chat_memory(model)
    return setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model,
                                       memory=memory, custom_prompt=custom_prompt, storage_context=storage_context)

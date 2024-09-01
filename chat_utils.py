from llama_index.core import SimpleDirectoryReader
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from utils import (setup_index_and_chat_engine, get_embedding_model, set_chat_memory,
                   set_ollama_llm, set_huggingface_llm, set_nvidia_model, set_openai_model, set_anth_model)
import torch, os, glob, gc, dotenv
dotenv.load_dotenv()

DIRECTORY_PATH = "data"
EMBED_MODEL = get_embedding_model()


# TODO Add parsing for different types of files. PDF, EXCEL FILES, etc..
def load_docs():
    all_files = glob.glob(os.path.join(DIRECTORY_PATH, "**", "*"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    documents = []
    for file in all_files:
        reader = SimpleDirectoryReader(input_files=[file]).load_data()
        documents.extend(reader)
    return documents

def load_github_repo(owner, repo, branch):
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
    ).load_data(branch=branch)
    return documents


def create_chat_engine(model_provider, model, temperature, max_tokens, custom_prompt, top_p,
                       context_window, quantization, owner, repo, branch):
    torch.cuda.empty_cache()
    gc.collect()
    documents = load_docs()
    # fix issue of model not getting context from repo ripper
    if owner and repo and branch:
        print(f"Loading GitHub repo: {owner}/{repo} (branch: {branch})")
        documents += load_github_repo(owner, repo, branch)
        if not documents:
            print(f"No documents loaded from {owner}/{repo} on branch {branch}")
        else:
            print(f"Loaded {len(documents)} documents from {owner}/{repo} on branch {branch}")
    else:
        documents = documents
    embed_model = EMBED_MODEL
    if model_provider == "Ollama":
        llm = set_ollama_llm(model, temperature, max_tokens)
    elif model_provider == "HuggingFace":
        llm = set_huggingface_llm(model, temperature, max_tokens, top_p, context_window, quantization)
    elif model_provider == "NVIDIA NIM":
        llm = set_nvidia_model(model, temperature, max_tokens, top_p)
    elif model_provider == "OpenAI":
        llm = set_openai_model(model, temperature, max_tokens, top_p, context_window)
    elif model_provider == "Anthropic":
        llm = set_anth_model(model, temperature, max_tokens, context_window)
    else:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    memory = set_chat_memory(model)
    return setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory,
                                       custom_prompt=custom_prompt)

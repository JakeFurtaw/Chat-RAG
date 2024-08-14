from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import torch

# Global variable to store the embedding model
_embed_model = None


def set_device(gpu: int = None) -> str:
    return f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"


def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_400M_v5", device=set_device(0),
                                            trust_remote_code=True)
    return _embed_model


def set_llm(model, temperature):
    llm_models = {
        "codestral:latest": {"model": "codestral:latest", "device": set_device(1)},
        "mistral-nemo:latest": {"model": "mistral-nemo:latest", "device": set_device(1)},
        "llama3.1:latest": {"model": "llama3.1:latest", "device": set_device(1)},
        "deepseek-coder-v2:latest": {"model": "deepseek-coder-v2:latest", "device": set_device(1)},
        "gemma2:latest": {"model": "gemma2:latest", "device": set_device(1)},
        "codegemma:latest": {"model": "codegemma:latest", "device": set_device(1)}
    }

    llm_config = llm_models.get(model, llm_models["codestral:latest"])
    return Ollama(model=llm_config["model"], request_timeout=30.0, device=llm_config["device"],
                  temperature=temperature, num_output=100, sream=True)


def set_chat_memory(model):
    memory_limits = {
        "codestral:latest": 30000,
        "mistral-nemo:latest": 115000,
        "llama3.1:latest": 115000,
        "deepseek-coder-v2:latest": 115000
    }
    token_limit = memory_limits.get(model, 6000)
    return ChatMemoryBuffer.from_defaults(token_limit=token_limit)


def setup_index_and_chat_engine(docs, embed_model, llm, memory):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    chat_prompt = (
        "You are an AI coding assistant, your primary function is to help users with\n"
        "coding-related questions and tasks. You have access to a knowledge base of programming documentation and\n"
        "best practices. When answering questions please follow these guidelines. 1. Provide clear, concise, and\n"
        "accurate code snippets when appropriate. 2. Explain your code and reasoning step by step. 3. Offer\n"
        "suggestions for best practices and potential optimizations. 4. If the user's question is unclear,\n"
        "ask for clarification dont assume or guess the answer to any question. 5. When referencing external \n"
        "libraries or frameworks, briefly explain their purpose. 6. If the question involves multiple possible\n"
        "approaches, outline the pros and cons of each.\n"
        "Response:"
    )

    system_message = ChatMessage(role="system", content=chat_prompt)
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        stream=True,
        system_prompt=system_message,
        llm=llm,
        context_prompt=("Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information above I want you to think step by step to answer \n"
                        "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: ")
    )
    return chat_engine

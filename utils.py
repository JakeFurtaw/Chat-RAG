from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import torch


def set_device(gpu: int = None) -> str:
    if torch.cuda.is_available() and gpu is not None:
        device = f"cuda:{gpu}"
    else:
        device = "cpu"
    return device


def set_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_400M_v5", device=set_device(0),
                                       trust_remote_code=True)
    return embed_model


def set_llm():
    llm = Ollama(model="codestral:latest", request_timeout=30.0, device=set_device(1))
    return llm


def setup_index_and_chat_engine(docs, embed_model, llm):
    memory = ChatMemoryBuffer.from_defaults(token_limit=6000)
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    # Define the chat prompt
    chat_prompt = (
        "You are an AI coding assistant powered by the Codestral model. Your primary function is to help users with\n"
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

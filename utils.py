from llama_index.core.chat_engine.types import ChatMode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage
from transformers import BitsAndBytesConfig
import torch, dotenv, os
from huggingface_hub import login

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def set_device(gpu: int = None) -> str:
    return f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"

def get_embedding_model():
    embed_model = HuggingFaceEmbedding(model_name="/home/jake/Programming/Models/embedding/stella_en_400M_v5",
                                       device=set_device(0), trust_remote_code=True)
    return embed_model

def set_ollama_llm(model, temperature, max_tokens):
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
                  temperature=temperature, additional_kwargs={"num_predict": max_tokens})

def set_huggingface_llm(model, temperature, max_tokens, top_p, context_window, quantization):
    if quantization == "2 Bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    elif quantization == "4 Bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization == "8 Bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8"
        )
    elif quantization == "No Quantization":
        quantization_config = None
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    model_kwargs = {"quantization_config": quantization_config}
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    return HuggingFaceLLM(
        model_name=model,
        tokenizer_name=model,
        context_window=context_window,
        max_new_tokens=max_tokens,
        model_kwargs=model_kwargs,
        is_chat_model= True,
        device_map="cuda:0",
        generate_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True
        },
    )

def set_nvidia_model(model, temperature, max_tokens, top_p):
    return NVIDIA(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        nvidia_api_key= os.getenv("NVIDIA_API_KEY")
    )

def set_chat_memory(model):
    memory_limits = {
        "codestral:latest": 30000,
        "mistralai/Codestral-22B-v0.1":30000,
        "mistral-nemo:latest": 124000,
        "mistralai/Mistral-Nemo-Instruct-2407": 124000,
        "llama3.1:latest": 124000,
        "meta-llama/Meta-Llama-3.1-8B-Instruct": 124000,
        "deepseek-coder-v2:latest": 124000,
        "deepseek-ai/DeepSeek-Coder-V2-Instruct": 124000
    }
    token_limit = memory_limits.get(model, 6000)
    return ChatMemoryBuffer.from_defaults(token_limit=token_limit)

def setup_index_and_chat_engine(docs, embed_model, llm, memory, custom_prompt):
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    Settings.llm = llm
    chat_prompt = (
        "You are an AI coding assistant, your primary function is to help users with coding-related questions \n"
        "and tasks. You have access to a knowledge base of programming documentation and best practices. \n"
        "When answering questions please follow these guidelines. 1. Provide clear, concise, and \n"
        "accurate code snippets when appropriate. 2. Explain your code and reasoning step by step. 3. Offer \n"
        "suggestions for best practices and potential optimizations. 4. If the user's question is unclear, \n"
        "ask for clarification dont assume or guess the answer to any question. 5. When referencing external \n"
        "libraries or frameworks, briefly explain their purpose. 6. If the question involves multiple possible \n"
        "approaches, outline the pros and cons of each. Always Remember to be friendly! \n"
        "Response:"
    )

    system_message = ChatMessage(role="system", content=chat_prompt if custom_prompt is None else custom_prompt)
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
                        "the query in a crisp manner, incase case you don't know the answer say 'I don't know!'. \n"
                        "Query: {query_str} \n"
                        "Answer: ")
    )
    return chat_engine

import torch, gc
import gradio as gr
from chat_utils import create_chat_engine
from config import HF_MODEL_LIST, OLLAMA_MODEL_LIST, NV_MODEL_LIST, OA_MODEL_LIST, ANTH_MODEL_LIST


# TODO needs to be optimized doesnt fully do its job.
# Clears gpu memory so a new model can be load.
def reset_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


"""
Main model class that deals with most of the functionality of the model and chat engine. This class handles the 
setting and resetting of the chat engine, model and model provider switching, database loading and resetting, and github
repository loading and reset/
"""
class ModelManager:
    def __init__(self):
        self.collection_name = None
        self.url = None
        self.password = None
        self.username = None
        self.vector_store = None
        self.model_param_updates = ModelParamUpdates(self)
        self.branch = None
        self.repo = None
        self.owner = None
        self.chat_engine = None
        self.provider = "Ollama"
        self.selected_model = "codestral:latest"
        self.model_display_names = {
            "Ollama": OLLAMA_MODEL_LIST,
            "HuggingFace": HF_MODEL_LIST,
            "NVIDIA NIM": NV_MODEL_LIST,
            "OpenAI": OA_MODEL_LIST,
            "Anthropic": ANTH_MODEL_LIST
        }

    # Creates the initial chat engine
    def create_initial_chat_engine(self):
        return create_chat_engine(self.provider, self.selected_model,
                                  self.model_param_updates.temperature,
                                  self.model_param_updates.max_tokens,
                                  self.model_param_updates.custom_prompt,
                                  self.model_param_updates.top_p,
                                  self.model_param_updates.context_window,
                                  self.model_param_updates.quantization,
                                  self.owner, self.repo, self.branch, self.vector_store, self.username,
                                  self.password, self.url, self.collection_name)

    # Processes the query from the user and sends it to the chat engine for processing
    def process_input(self, message):
        if self.chat_engine is None:
            self.chat_engine = self.create_initial_chat_engine()
        return self.chat_engine.stream_chat(message)

    # Updates the model provider and sends it to the chat engine based off the selection of the user
    def update_model_provider(self, provider):
        reset_gpu_memory()
        self.provider = provider
        default_models = {
            "Ollama": "codestral:latest",
            "HuggingFace": "",
            "NVIDIA NIM": "mistralai/codestral-22b-instruct-v0.1",
            "OpenAI": "gpt-4o",
            "Anthropic": "claude-3-5-sonnet-20240620"
        }
        self.selected_model = default_models.get(provider, "codestral:latest")
        gr.Info(f"Model provider updated to {provider}.", duration=10)
        self.reset_chat_engine()

    # Updates the model and sends it to the chat engine based off the selection of the user
    def update_model(self, display_name):
        reset_gpu_memory()
        self.selected_model = self.model_display_names[self.provider].get(display_name, self.selected_model)
        self.reset_chat_engine()
        gr.Info(f"Model updated to {display_name}.", duration=10)

    # Sets GitHub info to add its data to the context of the model
    def set_github_info(self, owner, repo, branch):
        self.owner, self.repo, self.branch = owner, repo, branch
        if all([owner, repo, branch]) != "":
            gr.Info(
                f"GitHub repository info set to Owners Username: {owner}, Repository Name: {repo}, and Branch Name: {branch}.")
        self.reset_chat_engine()

    # Resets GitHub info to remove the data from the context of the model
    def reset_github_info(self):
        self.owner = self.repo = self.branch = ""
        self.set_github_info(self.owner, self.repo, self.branch)
        gr.Info("GitHub repository info cleared and repository files from the models context!")
        self.reset_chat_engine()
        return self.owner, self.repo, self.branch

    # Sets database parameters and adds it to the models context
    def setup_database(self, vector_store, username, password, url, collection_name):
        self.vector_store, self.username, self.password, self.url, self.collection_name = vector_store, username, password, url, collection_name
        self.reset_chat_engine()
        gr.Info(f"Database connection established with {vector_store}.", duration=10)
        return self.vector_store

    # Resets database parameters and removes it from the models context
    def remove_database(self):
        self.vector_store = self.username = self.password = self.url = self.collection_name = None
        self.reset_chat_engine()
        gr.Info("Database connection removed.", duration=10)
        return self.username, self.password, self.url

    # Resets chat engine so the new parameters and new data can be loaded or removed into or from  the model
    def reset_chat_engine(self):
        reset_gpu_memory()
        self.chat_engine = self.create_initial_chat_engine()


"""
Secondary class that sets initial model and chat engine parameters as well as updates model and chat engine parameters.
It is also responsible for send gradio warning and info messages to the front end for the user to know their action was
a success.
"""
class ModelParamUpdates:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.max_tokens = 2048
        self.temperature = .75
        self.top_p = .4
        self.context_window = 2048
        self.quantization = "4 Bit"
        self.custom_prompt = None

    # Updates model quantization and sends a message to the user about the change. Also reset the gpu memory.
    def update_quant(self, quantization):
        reset_gpu_memory()
        self.quantization = quantization
        gr.Info(f"Quantization updated to {quantization}.", duration=10)
        self.model_manager.reset_chat_engine()

    # Updates model temperature parameter, send user a message about the change, and resets gpu memory.
    def update_model_temp(self, temperature):
        reset_gpu_memory()
        self.temperature = temperature
        gr.Info(f"Model temperature updated to {temperature}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   "and diversity of generated responses. Use with caution!",
                   duration=10)
        self.model_manager.reset_chat_engine()

    # Updates model top p parameter, send user a message about the change, and resets gpu memory.
    def update_top_p(self, top_p):
        reset_gpu_memory()
        self.top_p = top_p
        gr.Info(f"Top P updated to {top_p}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   "and diversity of generated responses. Use with caution!",
                   duration=10)
        self.model_manager.reset_chat_engine()

    # Updates model context window parameter, send user a message about the change, and resets gpu memory.
    def update_context_window(self, context_window):
        reset_gpu_memory()
        self.context_window = context_window
        gr.Info(f"Context Window updated to {context_window}.", duration=10)
        gr.Warning("Changing this value can affect the amount of the context the model can see and use "
                   "to answer your question.",
                   duration=10)
        self.model_manager.reset_chat_engine()

    # Updates the max output tokens a model can respond with send user a message about the change, and resets gpu memory.
    def update_max_tokens(self, max_tokens):
        reset_gpu_memory()
        self.max_tokens = max_tokens
        gr.Info(f"Max Tokens set to {max_tokens}.", duration=10)
        gr.Warning("Please note that reducing the maximum number of tokens may"
                   " cause incomplete or unexpected responses from the model if a user's question requires more tokens"
                   " for an accurate answer.",
                   duration=10)
        self.model_manager.reset_chat_engine()

    # Updates the chat engines system prompt, send user a message about the change, and resets gpu memory.
    def update_chat_prompt(self, custom_prompt):
        reset_gpu_memory()
        self.custom_prompt = custom_prompt
        gr.Warning("Caution: Changing the chat prompt may significantly alter the model's responses and could "
                   "potentially cause misleading or incorrect information to be generated. Please ensure that "
                   "the modified prompt is appropriate for your intended use case.", duration=10)
        self.model_manager.reset_chat_engine()

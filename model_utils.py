import torch, gc
import gradio as gr
from chat_utils import create_chat_engine
from config import HF_MODEL_LIST, OLLAMA_MODEL_LIST, NV_MODEL_LIST, OA_MODEL_LIST, ANTH_MODEL_LIST


def reset_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


class ModelManager:
    def __init__(self):
        self.model_param_updates = ModelParamUpdates(self)
        self.branch = None
        self.repo = None
        self.owner = None
        self.chat_engine = None
        self.provider = "Ollama"
        self.selected_model = "codestral:latest"
        self.ollama_model_display_names = OLLAMA_MODEL_LIST
        self.hf_model_display_names = HF_MODEL_LIST
        self.nv_model_display_names = NV_MODEL_LIST
        self.openai_model_display_names = OA_MODEL_LIST
        self.anth_model_display_names = ANTH_MODEL_LIST

    def create_initial_chat_engine(self):
        return create_chat_engine(self.provider, self.selected_model,
                                  self.model_param_updates.temperature,
                                  self.model_param_updates.max_tokens,
                                  self.model_param_updates.custom_prompt,
                                  self.model_param_updates.top_p,
                                  self.model_param_updates.context_window,
                                  self.model_param_updates.quantization,
                                  self.owner, self.repo, self.branch)

    def process_input(self, message):
        if self.chat_engine is None:
            self.chat_engine=self.create_initial_chat_engine()
        return self.chat_engine.stream_chat(message)

    def update_model_provider(self, provider):
        reset_gpu_memory()
        self.provider = provider
        if provider == "Ollama":
            self.selected_model = "codestral:latest"
            gr.Info(f"Model provider updated to {provider}.", duration=10)
        elif provider == "HuggingFace":
            gr.Info(f"Model provider updated to {provider}.", duration=10)
            gr.Warning("Model is loading, this could take some time depending on your hardware.", duration=30)
            self.selected_model = "mistralai/Codestral-22B-v0.1"
        elif provider == "NVIDIA NIM":
            gr.Info(f"Model provider updated to {provider}.", duration=10)
            self.selected_model = "mistralai/codestral-22b-instruct-v0.1"
        elif provider == "OpenAI":
            gr.Info(f"Model provider updated to {provider}.", duration=10)
            self.selected_model = "gpt-4o"
        elif provider == "Anthropic":
            gr.Info(f"Model provider updated to {provider}.", duration=10)
            self.selected_model = "claude-3-5-sonnet-20240620"
        self.reset_chat_engine()

    def update_model(self, display_name):
        if self.provider == "Ollama":
            self.selected_model = self.ollama_model_display_names.get(display_name, "codestral:latest")
        elif self.provider == "HuggingFace":
            reset_gpu_memory()
            self.selected_model = self.hf_model_display_names.get(display_name, "mistralai/Codestral-22B-v0.1")
        elif self.provider == "NVIDIA NIM":
            self.selected_model = self.nv_model_display_names.get(display_name, "mistralai/codestral-22b-instruct-v0.1")
        elif self.provider == "OpenAI":
            self.selected_model = self.openai_model_display_names.get(display_name, "gpt-4o")
        elif self.provider == "Anthropic":
            self.selected_model = self.openai_model_display_names.get(display_name, "claude-3-5-sonnet-20240620")
        self.reset_chat_engine()
        gr.Info(f"Model updated to {display_name}.", duration=10)

    def set_github_info(self, owner, repo, branch):
        self.owner = owner
        self.repo = repo
        self.branch = branch
        if repo and owner and branch != "":
            gr.Info(f"GitHub repository info set to Owners Username: {owner}, Repository Name: {repo}, and Branch Name: {branch}.")
        self.reset_chat_engine()

    def reset_github_info(self):
        self.owner = ""
        self.repo = ""
        self.branch = ""
        self.set_github_info(self.owner, self.repo, self.branch)
        gr.Info(f"GitHub repository info cleared and repository files from the models context!")
        self.reset_chat_engine()
        return self.owner, self.repo, self.branch


    def reset_chat_engine(self):
        reset_gpu_memory()
        self.chat_engine = create_chat_engine(self.provider, self.selected_model, self.model_param_updates.temperature,
                                              self.model_param_updates.max_tokens, self.model_param_updates.custom_prompt,
                                              self.model_param_updates.top_p, self.model_param_updates.context_window,
                                              self.model_param_updates.quantization, self.owner, self.repo, self.branch)

class ModelParamUpdates:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.max_tokens = 2048
        self.temperature = .75
        self.top_p = .4
        self.context_window = 2048
        self.quantization = "4 Bit"
        self.custom_prompt = None

    def update_quant(self, quantization):
        reset_gpu_memory()
        self.quantization = quantization
        gr.Info(f"Quantization updated to {quantization}.", duration=10)
        self.model_manager.reset_chat_engine()

    def update_model_temp(self, temperature):
        reset_gpu_memory()
        self.temperature = temperature
        gr.Info(f"Model temperature updated to {temperature}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   "and diversity of generated responses. Use with caution!",
                   duration=10)
        self.model_manager.reset_chat_engine()

    def update_top_p(self, top_p):
        reset_gpu_memory()
        self.top_p = top_p
        gr.Info(f"Top P updated to {top_p}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   "and diversity of generated responses. Use with caution!",
                   duration=10)
        self.model_manager.reset_chat_engine()

    def update_context_window(self, context_window):
        reset_gpu_memory()
        self.context_window = context_window
        gr.Info(f"Context Window updated to {context_window}.", duration=10)
        gr.Warning("Changing this value can affect the amount of the context the model can see and use "
                   "to answer your question.",
                   duration=10)
        self.model_manager.reset_chat_engine()

    def update_max_tokens(self, max_tokens):
        reset_gpu_memory()
        self.max_tokens = max_tokens
        gr.Info(f"Max Tokens set to {max_tokens}.", duration=10)
        gr.Warning("Please note that reducing the maximum number of tokens may"
                   " cause incomplete or unexpected responses from the model if a user's question requires more tokens"
                   " for an accurate answer.",
                   duration=10)
        self.model_manager.reset_chat_engine()

    def update_chat_prompt(self, custom_prompt):
        reset_gpu_memory()
        self.custom_prompt = custom_prompt
        gr.Warning("Caution: Changing the chat prompt may significantly alter the model's responses and could "
                   "potentially cause misleading or incorrect information to be generated. Please ensure that "
                   "the modified prompt is appropriate for your intended use case.", duration=10)
        self.model_manager.reset_chat_engine()

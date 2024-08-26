import dotenv
from chat_utils import create_chat_engine
import gradio as gr

dotenv.load_dotenv()


class ModelManager:
    def __init__(self):
        self.chat_engine = None
        self.custom_prompt = None
        self.model_provider = "Ollama"
        self.selected_model = "codestral:latest"
        self.max_tokens = 2048
        self.model_temp = .75
        self.model_top_p = .4
        self.quantization = "4 Bit"
        self.ollama_model_display_names = {
            "Codestral 22B": "codestral:latest",
            "Mistral-Nemo 12B": "mistral-nemo:latest",
            "Llama3.1 8B": "llama3.1:latest",
            "DeepSeek Coder V2 16B": "deepseek-coder-v2:latest",
            "Gemma2 9B": "gemma2:latest",
            "CodeGemma 7B": "codegemma:latest"
        }
        self.hf_model_display_names = {
            "Codestral 22B": "mistralai/Codestral-22B-v0.1",
            "Mistral-Nemo 12B-Instruct": "mistralai/Mistral-Nemo-Instruct-2407",
            "Llama3.1 8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "DeepSeek Coder V2 16B": "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            "Gemma2 9B": "mistralai/Mistral-Nemo-Instruct-2407",
            "CodeGemma 7B-Instruct": "google/codegemma-7b-it",
        }
        # TODO Add NVIDIA NIMS Model Names
        self.reset_chat_engine()

    def process_input(self, message):
        try:
            return self.chat_engine.stream_chat(message)
        except Exception as e:
            return f"Error: {str(e)}"
    #TODO Add NVIDIA NIMS
    def update_model(self, display_name):
        if self.model_provider == "Ollama":
            self.selected_model = self.ollama_model_display_names.get(display_name, "codestral:latest")
        elif self.model_provider == "Hugging Face":
            self.selected_model = self.hf_model_display_names.get(display_name, "mistralai/Codestral-22B-v0.1")
        else:
            self.selected_model = "codestral:latest"  # Default to Ollama model
        self.reset_chat_engine()
        gr.Info(f"Model updated to {display_name}.", duration=10)
    #TODO Add NVIDIA NIMS
    def update_model_provider(self, provider):
        self.model_provider = provider
        if provider == "Ollama":
            self.selected_model = "codestral:latest"
        elif provider == "Hugging Face":
            self.selected_model = "mistralai/Codestral-22B-v0.1"
        self.reset_chat_engine()
        gr.Info(f"Model provider updated to {provider}.", duration=10)

    def update_model_temp(self, temperature):
        self.model_temp = temperature
        gr.Info(f"Model temperature updated to {temperature}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   f"and diversity of generated responses. Use with caution!",
                   duration=10)
        self.reset_chat_engine()

    def update_top_p(self, top_p):
        self.model_top_p = top_p
        gr.Info(f"Top P updated to {top_p}.", duration=10)
        gr.Warning("Changing this value can affect the randomness "
                   f"and diversity of generated responses. Use with caution!",
                   duration=10)
        self.reset_chat_engine()

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        gr.Info(f"Max Tokens set to {max_tokens}.", duration=10)
        gr.Warning( "Please note that reducing the maximum number of tokens may"
                   f" cause incomplete or unexpected responses from the model if a user's question requires more tokens"
                   f" for an accurate answer.",
                   duration=10)
        self.reset_chat_engine()

    def update_chat_prompt(self, custom_prompt):
        self.custom_prompt = custom_prompt
        gr.Warning("Caution: Changing the chat prompt may significantly alter the model's responses and could "
                   "potentially cause misleading or incorrect information to be generated. Please ensure that "
                   "the modified prompt is appropriate for your intended use case.", duration=10)
        self.reset_chat_engine()

    def reset_chat_engine(self):
        self.chat_engine = create_chat_engine(self.model_provider, self.selected_model, self.model_temp,
                                              self.max_tokens, self.custom_prompt, self.model_top_p)

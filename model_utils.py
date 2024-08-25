import dotenv
from chat_utils import create_chat_engine
import gradio as gr
dotenv.load_dotenv()


class ModelManager:
    def __init__(self):
        self.custom_prompt = None
        self.selected_model = "codestral:latest"
        self.max_tokens = 2048
        self.model_temp = .75
        self.chat_engine = create_chat_engine(self.selected_model, self.model_temp, self.max_tokens, self.custom_prompt)
        self.ollama_model_display_names = {
            "Codestral 22B": "codestral:latest",
            "Mistral-Nemo 12B": "mistral-nemo:latest",
            "Llama3.1 8B": "llama3.1:latest",
            "DeepSeek Coder V2 16B": "deepseek-coder-v2:latest",
            "Gemma2 9B": "gemma2:latest",
            "CodeGemma 7B": "codegemma:latest"
        }

    def process_input(self, message):
        try:
            return self.chat_engine.stream_chat(message)
        except Exception as e:
            return f"Error: {str(e)}"

    def update_model(self, display_name):
        model_name = self.ollama_model_display_names.get(display_name, "codestral:latest")
        self.selected_model = model_name
        self.reset_chat_engine()
        gr.Warning(f"Model updated to {display_name}. Please make sure you have this model installed "
                   f"through Ollama!", duration=10)

    def update_model_temp(self, temperature):
        self.model_temp = temperature
        self.reset_chat_engine()

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        gr.Warning("WARNING: This may cut the output of the model short if your response requires more tokens "
                   "for the answer!!!", duration=10)
        self.reset_chat_engine()

    def update_chat_prompt(self, custom_prompt):
        self.custom_prompt=custom_prompt
        gr.Warning("WARNING: Changing the custom prompt may affect the model's responses. Use this feature with "
                   "caution.", duration=10)
        self.reset_chat_engine()

    def reset_chat_engine(self):
        self.chat_engine = create_chat_engine(self.selected_model, self.model_temp, self.max_tokens, self.custom_prompt)

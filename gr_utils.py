import os
import shutil
import gradio as gr
from model_utils import ModelManager
import logging


class GRUtils:
    def __init__(self):
        self.model_temp = .75
        self.max_tokens = 2048
        self.model_manager = ModelManager()
        self.chat_history = []
        self.model_display_names = {
            "Codestral 22B": "codestral:latest", "Mistral-Nemo 12B": "mistral-nemo:latest",
            "Llama3.1 8B": "llama3.1:latest", "DeepSeek Coder V2 16B": "deepseek-coder-v2:latest",
            "Gemma2 9B": "gemma2:latest", "CodeGemma 7B": "codegemma:latest"
        }

    def chat(self, message: str):
        response = self.model_manager.process_input(message)
        self.chat_history.append((message, str(response)))
        return "", self.chat_history

    def stream_response(self, message: str):
        streaming_response = self.model_manager.process_input(message)
        full_response = ""
        for delta in streaming_response.response_gen:
            if isinstance(delta, str):
                full_response += delta
            else:
                full_response += delta.delta
            yield "", self.chat_history + [(message, full_response)]
        self.chat_history.append((message, full_response))

    def clear_chat_history(self):
        self.chat_history.clear()

    def clear_his_and_mem(self):
        self.clear_chat_history()
        self.model_manager.reset_chat_engine()

    def delete_db(self):
        gr.Warning("Wait about 5-10 seconds for the files to clear. After this message disappears you should  "
                   "be in the clear.", duration=10)
        if os.path.exists("data"):
            shutil.rmtree("data")
            os.makedirs("data")
        self.model_manager.reset_chat_engine()

    def update_max_tokens(self, max_tokens):
        self.model_manager.update_max_tokens(max_tokens)
        gr.Warning("WARNING: This may cut the output of the model short if your response requires more tokens "
                   "for the answer!!!", duration=10)

    def update_model_temp(self, temperature):
        self.model_manager.update_temperature(temperature)

    def update_chat_prompt(self, custom_prompt):
        self.model_manager.update_chat_prompt(custom_prompt)
        gr.Warning("WARNING: Changing the custom prompt may affect the model's responses. Use this feature with "
                   "caution.", duration=10)

    def update_model(self, display_name):
        model_name = self.model_display_names.get(display_name, "codestral:latest")
        self.model_manager.update_model(model_name)
        self.chat_history.clear()
        gr.Warning(f"Model updated to {display_name}. Please make sure you have this model installed through "
                   "Ollama!", duration=10)

    @staticmethod
    async def handle_doc_upload(files):
        gr.Warning("Make sure you hit the upload button or the model wont see your files!", duration=10)
        return [file.name for file in files]
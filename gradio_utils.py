import os
import shutil
import gradio as gr
from model_utils import ModelManager

class GradioUtils:
    def __init__(self):
        self.model_manager = ModelManager()
        self.chat_history = []

    def stream_response(self, message: str):
        streaming_response = self.model_manager.process_input(message)
        full_response = ""
        for delta in streaming_response.response_gen:
            full_response += delta
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

    def update_model_temp(self, temperature):
        self.model_manager.update_model_temp(temperature)

    def update_chat_prompt(self, custom_prompt):
        self.model_manager.update_chat_prompt(custom_prompt)

    def update_model(self, display_name):
        self.clear_chat_history()
        self.model_manager.update_model(display_name)

    @staticmethod
    async def handle_doc_upload(files):
        gr.Warning("Make sure you hit the upload button or the model wont see your files!", duration=10)
        return [file.name for file in files]
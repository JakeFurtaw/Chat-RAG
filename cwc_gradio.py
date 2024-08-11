import os
import shutil

import gradio as gr
from model_manager import ModelManager


def delete_kb():
    for item_name in os.listdir("data"):
        item_path = os.path.join("data", item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error deleting {item_path}: {e}")


class CWCGradio:
    def __init__(self):
        self.model_temp = .75
        self.file_paths = None
        self.model_manager = ModelManager()
        self.chat_history = []

    def chat(self, message):
        response = self.model_manager.process_input(message)
        self.chat_history.append((message, response))
        return "", self.chat_history

    def load_chat_history(self):

        return self.chat_history

    def clear_chat_history(self):
        self.chat_history.clear()
        return self.chat_history

    def clear_his_and_mem(self):
        self.clear_chat_history()
        self.model_manager.reset_chat_engine()
        return []

    def update_model_temp(self, temperature):
        self.model_temp = temperature
        self.model_manager.reset_chat_engine()
        return self.model_temp

    def update_model(self, model):
        self.model_manager.update_model(model)
        self.chat_history.clear()

    def handle_doc_upload(self, files):
        self.file_paths = [file.name for file in files]
        self.model_manager.reset_chat_engine()
        return self.file_paths

    def launch(self):
        with gr.Blocks(title="Chat RAG", theme="monochrome", fill_height=True, fill_width=True) as iface:
            gr.Markdown("# Chat RAG: Interactive Coding Assistant")
            gr.Markdown("### This app is a chat-based coding assistant with a graphical user interface built using "
                        "Gradio. It allows users to interact with various language models to ask coding questions, "
                        "with the ability to upload files for additional context. The app utilizes RAG ("
                        "Retrieval-Augmented Generation) to provide more informed responses based on the loaded "
                        "documents and user queries.")
            with gr.Row():
                with gr.Column(scale=8, variant="compact"):
                    chatbot = gr.Chatbot(label="Chat RAG", container=False, show_copy_button=True, height=600)
                    msg = gr.Textbox(show_label=False, autoscroll=True, autofocus=True, container=False,
                                     placeholder="Enter your coding question here...")
                    with gr.Row():
                        load_chat_history = gr.Button(value="Load Chat History")
                        clear = gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                        clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
                    msg.submit(self.chat, inputs=[msg], outputs=[msg, chatbot], show_progress="full")

                with gr.Column(scale=2):
                    gr.Markdown("### Upload File(s) before querying the chatbot, otherwise the files wont be seen by "
                                "the model.")
                    files = gr.Files(interactive=True, label="Upload Files Here", container=False,
                                     file_count="multiple", file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c"
                                                                                                                ".css",
                                                                        ".cpp", ".html", ".docx", ".doc", ".js",
                                                                        ".jsx", ".xml"])
                    clear_kb = gr.Button(value="Clear RAG Database", interactive=True)
                    temperature = gr.Slider(minimum=.1, maximum=1, value=.75, label="Model Temperature",
                                            info="Select a temperature between .1 and 1 to set the model to.",
                                            interactive=True, step=.05)
                    temp_state = gr.State(value=.75)
                    selected_chat_model = gr.Dropdown(
                        choices=["codestral:latest", "mistral-nemo:latest", "llama3.1:latest",
                                 "deepseek-coder-v2:latest", "gemma2:latest", "codegemma:latest"],
                        label="Select Chat Model", value="codestral:latest", interactive=True, filterable=True,
                        info="Choose the model you want to use from the list below.")

                # Left Column Button Functionally
                load_chat_history.click(self.load_chat_history, outputs=chatbot)
                clear.click(self.clear_chat_history, outputs=chatbot)
                clear_chat_mem.click(self.clear_his_and_mem, outputs=chatbot)

                # Right Column Button Functionally
                files.upload(self.handle_doc_upload)
                clear_kb.click(delete_kb)
                temperature.release(self.update_model_temp, inputs=[temperature], outputs=[temp_state])
                selected_chat_model.change(self.update_model, inputs=selected_chat_model)

        iface.launch(inbrowser=True)

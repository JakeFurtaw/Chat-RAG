import os
import shutil
import gradio as gr
from model_utils import ModelManager


def delete_kb():
    if os.path.exists("data"):
        shutil.rmtree("data")
        os.makedirs("data")


def kb_warning():
    gr.Warning("Wait about 5-10 seconds for the files to clear. After this message disappears you should  "
               "be in the clear.", duration=7)


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

    def clear_chat_history(self):
        self.chat_history.clear()
        return self.chat_history

    def clear_his_and_mem(self):
        self.clear_chat_history()
        self.model_manager.reset_chat_engine()
        return []

    def upload_button(self):
        self.model_manager.reset_chat_engine()
        return []

    def update_model_temp(self, temperature):
        self.model_temp = temperature
        self.model_manager.update_temperature(temperature)
        return self.model_temp

    def update_model(self, model):
        self.model_manager.update_model(model)
        self.chat_history.clear()

    def handle_doc_upload(self, files):
        self.file_paths = [file.name for file in files]
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
                        clear = gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                        clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
                    msg.submit(self.chat, inputs=[msg], outputs=[msg, chatbot], show_progress="full")

                with gr.Column(scale=2):
                    gr.Markdown("### Upload File(s) before querying the chatbot, otherwise the files wont be seen by "
                                "the model.")
                    files = gr.Files(interactive=True, label="Upload Files Here", container=False,
                                     file_count="multiple", file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c"
                                                                        ".css", ".cpp", ".html", ".docx", ".doc", ".js",
                                                                        ".jsx", ".xml"])
                    with gr.Row():
                        upload = gr.Button(value="Upload Data", interactive=True)
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

                clear.click(self.clear_chat_history, outputs=chatbot)
                clear_chat_mem.click(self.clear_his_and_mem, outputs=chatbot)

                files.upload(self.handle_doc_upload)
                upload.click(self.upload_button)
                clear_kb.click(delete_kb)
                clear_kb.click(kb_warning)
                temperature.release(self.update_model_temp, inputs=[temperature], outputs=[temp_state])
                selected_chat_model.change(self.update_model, inputs=selected_chat_model)

        iface.launch(inbrowser=True, share=True)

import os
import shutil
import gradio as gr
from model_utils import ModelManager


class CWCGradio:
    def __init__(self):
        self.model_temp = .75
        self.max_tokens = 2048
        self.file_paths = None
        self.model_manager = ModelManager()
        self.chat_history = []
        self.model_display_names = {
            "Codestral 22B": "codestral:latest", "Mistral-Nemo 12B": "mistral-nemo:latest",
            "Llama3.1 8B": "llama3.1:latest", "DeepSeek Coder V2 16B": "deepseek-coder-v2:latest",
            "Gemma2 9B": "gemma2:latest", "CodeGemma 7B": "codegemma:latest"
        }

    def chat(self, message):
        response = self.model_manager.process_input(message)
        self.chat_history.append((message, str(response)))
        return "", self.chat_history

    def stream_response(self, message):
        streaming_response = self.model_manager.process_input(message)
        full_response = ""
        for delta in streaming_response.response_gen:
            full_response += delta
            yield "", self.chat_history + [(message, full_response)]
        self.chat_history.append((message, full_response))

    def clear_chat_history(self):
        self.chat_history.clear()
        return self.chat_history

    def clear_his_and_mem(self):
        self.clear_chat_history()
        self.model_manager.reset_chat_engine()

    def upload_button(self):
        self.model_manager.reset_chat_engine()

    def delete_db(self):
        gr.Warning("Wait about 5-10 seconds for the files to clear. After this message disappears you should  "
                   "be in the clear.", duration=10)
        if os.path.exists("data"):
            shutil.rmtree("data")
            os.makedirs("data")
        self.model_manager.reset_chat_engine()

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        self.model_manager.update_max_tokens(max_tokens)
        gr.Warning("WARNING: This may cut the output of the model short if your response requires more tokens "
                   "for the answer!!!", duration=10)
        return max_tokens

    def update_model_temp(self, temperature):
        self.model_temp = temperature
        self.model_manager.update_temperature(temperature)
        return self.model_temp

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        self.model_manager.update_max_tokens(max_tokens)
        gr.Warning("WARNING: This may cut the output of the model short if your response requires more tokens "
                   "for the answer!!!", duration=10)
        return max_tokens

    def update_model(self, display_name):
        model_name = self.model_display_names.get(display_name, "codestral:latest")
        self.model_manager.update_model(model_name)
        self.chat_history.clear()
        gr.Warning(f"Model updated to {display_name}. Please make sure you have this model installed through "
                   "Ollama!", duration=10)
        return []

    def handle_doc_upload(self, files):
        gr.Warning("Make sure you hit the upload button or the model wont see your files!", duration=10)
        self.file_paths = [file.name for file in files]
        return self.file_paths

    def launch(self):
        with gr.Blocks(title="Chat RAG", theme="monochrome", fill_height=True, fill_width=True) as iface:
            gr.Markdown("# Chat RAG: Interactive Coding Assistant"
                        "\nThis app is a chat-based coding assistant with a graphical user interface built using "
                        "Gradio. It allows users to interact with various language models to ask coding questions, "
                        "with the ability to upload files for additional context. "
                        "The app utilizes RAG (Retrieval-Augmented Generation) to provide more informed responses "
                        "based on the loaded documents and user queries.")
            with gr.Row():
                with gr.Column(scale=8, variant="compact"):
                    chatbot = gr.Chatbot(label="Chat RAG", container=False, show_copy_button=True, height=600)
                    msg = gr.Textbox(show_label=False, autoscroll=True, autofocus=True, container=False,
                                     placeholder="Enter your coding question here...")
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                        clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
                    msg.submit(self.stream_response, inputs=[msg], outputs=[msg, chatbot], show_progress="full")

                with gr.Column(scale=2):
                    files = gr.Files(interactive=True, label="Upload Files Here", container=False,
                                     file_count="multiple", file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c",
                                                                        ".css", ".cpp", ".html", ".docx", ".doc", ".js",
                                                                        ".jsx", ".xml"])
                    with gr.Row():
                        upload = gr.Button(value="Upload Data", interactive=True)
                        clear_db = gr.Button(value="Clear RAG Database", interactive=True)
                    temperature = gr.Slider(minimum=.1, maximum=1, value=.75, label="Model Temperature",
                                            info="Select a temperature between .1 and 1 to set the model to.",
                                            interactive=True, step=.05)
                    max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                                           label="Max Tokens in Response",
                                           info="Set the maximum number of tokens the model can respond with.")
                    temp_state = gr.State(value=.75)
                    selected_chat_model = gr.Dropdown(choices=list(self.model_display_names.keys()), interactive=True,
                                                      label="Select Chat Model", value="Codestral 22B", filterable=True,
                                                      info="Choose the model you want to chat with from the list below."
                                                      )
                # ---------Button Functionality controlled below----------------
                # Buttons in Left Column
                clear.click(self.clear_chat_history, outputs=chatbot)
                clear_chat_mem.click(self.clear_his_and_mem, outputs=chatbot)
                # Buttons in Right Column
                files.upload(self.handle_doc_upload, show_progress="full")
                upload.click(self.upload_button)
                clear_db.click(self.delete_db, show_progress="full")
                temperature.release(self.update_model_temp, inputs=[temperature], outputs=[temp_state])
                max_tokens.release(self.update_max_tokens, inputs=[max_tokens], outputs=[max_tokens])
                selected_chat_model.change(self.update_model, inputs=selected_chat_model, outputs=[chatbot])

        iface.launch(inbrowser=True, share=True)

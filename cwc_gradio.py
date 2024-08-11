import gradio as gr
from model_manager import ModelManager


class CWCGradio:
    def __init__(self):
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
        self.chat_history.clear()
        self.model_manager.reset_chat_engine()
        return []

    def update_model(self, model):
        self.model_manager.update_model(model)
        self.chat_history.clear()

    def handle_doc_upload(self, files):

        return []

    def launch(self):
        with gr.Blocks(title="Chat RAG", theme="monochrome", fill_height=True, fill_width=True) as iface:
            gr.Markdown("# Chat RAG: Interactive Coding Assistant")
            gr.Markdown("### This app is a chat-based coding assistant with a graphical user interface built using "
                        "Gradio. It allows users to interact with various language models to ask coding questions, "
                        "with the ability to upload files for additional context. \n### The app utilizes RAG ("
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

                with gr.Column(scale=1):
                    files = gr.Files(label="Upload Files Here")
                    upload_button = gr.UploadButton(label="Click to Upload Files")
                    selected_chat_model = gr.Dropdown(
                        choices=["codestral:latest", "mistral-nemo:latest", "llama3.1:latest",
                                 "deepseek-coder-v2:latest", "gemma2:latest", "codegemma:latest"],
                        label="Select Chat Model", value="codestral:latest", interactive=True, filterable=True,
                        info="Choose the model you want to use from the list below.")

                # Left Column Button Functionally
                load_chat_history.click(fn=self.load_chat_history, outputs=chatbot)
                clear.click(fn=self.clear_chat_history, outputs=chatbot)
                clear_chat_mem.click(fn=self.clear_his_and_mem, outputs=chatbot)

                # Right Column Button Functionally
                upload_button.upload(fn=self.handle_doc_upload, inputs=[upload_button], outputs=[files])
                selected_chat_model.change(fn=self.update_model, inputs=selected_chat_model)

        iface.launch(inbrowser=True, share=True)

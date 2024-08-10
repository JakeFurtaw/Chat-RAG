import gradio as gr

from chat import load_docs
from model_manager import ModelManager
from utils import setup_index_and_chat_engine, set_embedding_model, set_llm, set_chat_memory


class CWCGradio:

    def __init__(self):
        self.chat_engine = None
        self.model_manager = ModelManager()
        self.chat_history = []
        self.chat_memory = set_chat_memory(self.model_manager.selected_model)

    def chat(self, message):
        response = self.model_manager.process_input(message)
        self.chat_history.append((message, response))
        return "", self.chat_history

    # loads chat history
    def load_chat_history(self):
        loaded_history = self.chat_history
        return loaded_history

    # function clears the chat window
    def clear_chat_history(self):
        self.chat_history.clear()
        return self.chat_history

    # function clears the chat window and chat memory
    def clear_his_and_mem(self):
        self.chat_history.clear()
        self.chat_memory = set_chat_memory(self.model_manager.selected_model).reset()
        return []

    # update model gets the new model and clears chat history and memory
    def update_model(self, model):
        self.model_manager.update_model(model)
        self.chat_history.clear()
        self.chat_memory = set_chat_memory(self.model_manager.selected_model).reset()

    # reloading chat engine to grab new files when files are uploaded
    def update_chat_engine(self, files=None):
        documents = load_docs()  # Load documents from the directory
        embed_model = set_embedding_model()
        llm = set_llm(self.model_manager.selected_model)
        memory = set_chat_memory(self.model_manager.selected_model)
        self.chat_engine = setup_index_and_chat_engine(docs=documents, llm=llm, embed_model=embed_model, memory=memory)

    def launch(self):
        with gr.Blocks(title="Chat RAG", theme="monochrome", fill_height=True, fill_width=True) as iface:
            gr.Markdown("# Chat RAG: Interactive Coding Assistant")
            gr.Markdown("### This app is a chat-based coding assistant with a graphical user interface built using "
                        "Gradio. It allows users to interact with various language models to ask coding questions, "
                        "with the ability to upload files for additional context. \n### The app utilizes RAG ("
                        "Retrieval-Augmented Generation) to provide more informed responses based on the loaded "
                        "documents and user queries.")
            with gr.Row():
                # Left Column
                with gr.Column(scale=8, variant="compact"):
                    chatbot = gr.Chatbot(label="Chat RAG", container=False, show_copy_button=True, height=600)
                    msg = gr.Textbox(show_label=False, autoscroll=True, autofocus=True, container=False,
                                     placeholder="Enter your coding question here...")
                    with gr.Row():
                        load_chat_history = gr.Button(value="Load Chat History")
                        clear = gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                        clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
                    msg.submit(self.chat, inputs=[msg], outputs=[msg, chatbot], show_progress="full")

                # Right Column
                with gr.Column(scale=1):
                    files = gr.Files(label="Upload Your Files")
                    selected_chat_model = gr.Dropdown(
                        choices=["codestral:latest", "mistral-nemo:latest", "llama3.1:latest",
                                 "deepseek-coder-v2:latest", "gemma2:latest", "codegemma:latest"],
                        label="Select Chat Model", value="codestral:latest", interactive=True, filterable=True,
                        info="Choose the model you want to use from the list below.")

                # ---Button/Dropdown Functionality Implementations---
                # Left Column Buttons
                load_chat_history.click(fn=self.load_chat_history, outputs=chatbot)
                clear.click(fn=self.clear_chat_history, outputs=chatbot)
                clear_chat_mem.click(fn=self.clear_his_and_mem, outputs=chatbot)
                # Right Column Buttons
                selected_chat_model.change(fn=self.update_model, inputs=selected_chat_model)
                files.upload(fn=self.update_chat_engine, inputs=files)

        iface.launch(inbrowser=True, share=True)

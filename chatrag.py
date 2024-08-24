import gradio as gr
from gr_utils import GRUtils

grutils = GRUtils()

with gr.Blocks(title="Chat RAG", theme="monochrome", fill_height=True, fill_width=True) as demo:
    gr.Markdown("# Chat RAG: Interactive Coding Assistant"
                "\nThis app is a chat-based coding assistant with a graphical user interface built using "
                "Gradio. It allows users to interact with various language models to ask coding questions, "
                "with the ability to upload files for additional context. "
                "The app utilizes RAG (Retrieval-Augmented Generation) to provide more informed responses "
                "based on the loaded documents and user queries.")
    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Chat RAG",height=1000)
            msg = gr.Textbox(label="Textbox", placeholder="Enter your message here and hit return when you're ready...",
                             interactive=True)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot], value="Clear Chat Window")
                clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
            msg.submit(grutils.stream_response, inputs=[msg], outputs=[msg, chatbot], show_progress="full")
        with gr.Column(scale=2):
            files = gr.Files(interactive=True, label="Upload Files Here", file_count="multiple",
                             file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                         ".css", ".cpp", ".html", ".docx", ".doc", ".js"])
            with gr.Row():
                upload = gr.Button(value="Upload Data", interactive=True)
                clear_db = gr.Button(value="Clear RAG Database", interactive=True)
            gr.Radio(label="Select Model Provider",value="Ollama", choices=["Ollama", "HuggingFace", "NVIDIA NIM"],
                     interactive=True, info="Choose your model provider.")
            selected_chat_model = gr.Dropdown(choices=list(grutils.ollama_model_display_names.keys()),
                                              interactive=True,
                                              label="Select Chat Model", value="Codestral 22B",
                                              filterable=True,
                                              info="Choose the model you want to chat with from the list below."
                                              )
            temperature = gr.Slider(minimum=.1, maximum=1, value=.75, step=.05, label="Model Temperature",
                                    info="Select a temperature between .1 and 1 to set the model to.",
                                    interactive=True)
            max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1, label="Max Output Tokens",
                                   info="Set the maximum number of tokens the model can respond with.",
                                   interactive=True)
            custom_prompt = gr.Textbox(label="Enter a Custom Prompt", placeholder="Enter your custom prompt here..",
                                       interactive=True)
        # Button Functionality For RAG Chat
        # Buttons in Left Column
        selected_chat_model.change(grutils.update_model, inputs=selected_chat_model, outputs=[chatbot])
        clear.click(grutils.clear_chat_history, outputs=chatbot)
        clear_chat_mem.click(grutils.clear_his_and_mem, outputs=chatbot)
        # Buttons in Right Column
        files.upload(grutils.handle_doc_upload, show_progress="full")
        upload.click(lambda: grutils.model_manager.reset_chat_engine())
        clear_db.click(grutils.delete_db, show_progress="full")
        temperature.release(grutils.update_model_temp, inputs=[temperature])
        max_tokens.release(grutils.update_max_tokens, inputs=[max_tokens])
        custom_prompt.submit(grutils.update_chat_prompt, inputs=[custom_prompt])

demo.launch(inbrowser=True, share=True)

import gradio as gr
from gradio_utils import GradioUtils
from model_utils import ModelManager

gradioUtils = GradioUtils()
modelUtils = ModelManager()

# --------------Ollama Components------------------------
selected_chat_model = gr.Dropdown(choices=list(modelUtils.ollama_model_display_names.keys()),
                                  interactive=True,
                                  label="Select Chat Model",
                                  value="Codestral 22B",
                                  filterable=True,
                                  info="Choose the model you want to chat with from the list below.")
temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                        label="Model Temperature",
                        info="Select a temperature between 0 and 1 for the model.",
                        interactive=True)
max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                       label="Max Output Tokens",
                       info="Set the maximum number of tokens the model can respond with.",
                       interactive=True)
custom_prompt = gr.Textbox(label="Enter a Custom Prompt",
                           placeholder="Enter your custom prompt here...",
                           interactive=True)

# ------------------HuggingFace components-------------------------------
hf_model = gr.Dropdown(choices=["Model1", "Model2", "Model3"],
                       interactive=True,
                       label="Select Chat Model",
                       value="Model1",
                       filterable=True,
                       info="Choose a Hugging Face model.",
                       visible=False)
hf_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                           label="Model Temperature",
                           info="Select a temperature between 0 and 1 for the model.",
                           interactive=True,
                           visible=False)
hf_top_p = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05,
                     label="Top P",
                     info="Select a top p value between 0 and 1 for the model.",
                     interactive=True,
                     visible=False)
hf_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                          label="Max Output Tokens",
                          info="Set the maximum number of tokens the model can respond with.",
                          interactive=True,
                          visible=False)
hf_custom_prompt = gr.Textbox(label="Enter a Custom Prompt",
                              placeholder="Enter your custom prompt here...",
                              interactive=True,
                              visible=False)

# ----------------------------NVIDIA NIM components---------------------------
nim_model = gr.Dropdown(choices=["NIM1", "NIM2", "NIM3"],
                        interactive=True,
                        label="Select NVIDIA NIM Model",
                        value="NIM1",
                        filterable=True,
                        info="Choose a NVIDIA NIM model.",
                        visible=False)
nv_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                           label="Model Temperature",
                           info="Select a temperature between .1 and 1 to set the model to.",
                           interactive=True,
                           visible=False)
nv_top_p = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05,
                     label="Top P",
                     info="Set the top p value for the model.",
                     interactive=True,
                     visible=False)
nv_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                          label="Max Output Tokens",
                          info="Set the maximum number of tokens the model can respond with.",
                          interactive=True,
                          visible=False)

# ----------Gradio Layout-----------------------------
with gr.Blocks(title="Chat RAG",
               theme="monochrome",
               fill_height=True,
               fill_width=True) as demo:
    gr.Markdown("# Chat RAG: Interactive Coding Assistant"
                "\nThis app is a chat-based coding assistant with a graphical user interface built using "
                "Gradio. It allows users to interact with various language models to ask coding questions, "
                "with the ability to upload files for additional context. "
                "The app utilizes RAG (Retrieval-Augmented Generation) to provide more informed responses "
                "based on the loaded documents and user queries.")
    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(label="Chat RAG",height=1000)
            msg = gr.Textbox(label="Textbox",
                             placeholder="Enter your message here and hit return when you're ready...",
                             interactive=True)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot],
                                       value="Clear Chat Window")
                clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory")
        with gr.Column(scale=2):
            files = gr.Files(interactive=True,
                             label="Upload Files Here",
                             file_count="multiple",
                             file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                         ".css", ".cpp", ".html", ".docx", ".doc", ".js", ".json"])
            with gr.Row():
                upload = gr.Button(value="Upload Data",
                                   interactive=True)
                clear_db = gr.Button(value="Clear RAG Database",
                                     interactive=True)
            model_provider = gr.Radio(label="Select Model Provider",
                                      value="Ollama",
                                      choices=["Ollama", "Hugging Face", "NVIDIA NIM"],
                                      interactive=True,
                                      info="Choose your model provider.")

            selected_chat_model.render()
            temperature.render()
            max_tokens.render()
            custom_prompt.render()
            hf_model.render()
            hf_temperature.render()
            hf_top_p.render()
            hf_max_tokens.render()
            hf_custom_prompt.render()
            nim_model.render()
            nv_temperature.render()
            nv_top_p.render()
            nv_max_tokens.render()

            def update_layout(choice):
                ollama_visible = choice == "Ollama"
                hf_visible = choice == "Hugging Face"
                nim_visible = choice == "NVIDIA NIM"
                return (
                    gr.update(visible=ollama_visible),
                    gr.update(visible=ollama_visible),
                    gr.update(visible=ollama_visible),
                    gr.update(visible=ollama_visible),
                    gr.update(visible=hf_visible),
                    gr.update(visible=hf_visible),
                    gr.update(visible=hf_visible),
                    gr.update(visible=hf_visible),
                    gr.update(visible=hf_visible),
                    gr.update(visible=nim_visible),
                    gr.update(visible=nim_visible),
                    gr.update(visible=nim_visible),
                    gr.update(visible=nim_visible)
                )

            model_provider.change(
                fn=update_layout,
                inputs=[model_provider],
                outputs=[selected_chat_model, temperature, max_tokens, custom_prompt,
                         hf_model, hf_temperature, hf_top_p, hf_max_tokens, hf_custom_prompt,
                         nim_model, nv_temperature, nv_top_p, nv_max_tokens]
            )

# -------Button Functionality For RAG Chat-----------
        msg.submit(gradioUtils.stream_response, inputs=[msg], outputs=[msg, chatbot], show_progress="full")
        # Buttons in Left Column
        selected_chat_model.change(gradioUtils.update_model, inputs=selected_chat_model, outputs=[chatbot])
        clear.click(gradioUtils.clear_chat_history, outputs=chatbot)
        clear_chat_mem.click(gradioUtils.clear_his_and_mem, outputs=chatbot)
        # Buttons in Right Column
        files.upload(gradioUtils.handle_doc_upload, show_progress="full")
        upload.click(lambda: gradioUtils.model_manager.reset_chat_engine())
        clear_db.click(gradioUtils.delete_db, show_progress="full")
        temperature.release(gradioUtils.update_model_temp, inputs=[temperature])
        max_tokens.release(gradioUtils.update_max_tokens, inputs=[max_tokens])
        custom_prompt.submit(gradioUtils.update_chat_prompt, inputs=[custom_prompt])

demo.launch(inbrowser=True, share=True)

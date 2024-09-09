import os
import gradio as gr
from gradio_utils import GradioUtils
from model_utils import ModelManager
import dotenv
dotenv.load_dotenv()
gradioUtils = GradioUtils()
modelUtils = ModelManager()

css = """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
#button{
background:#06354d
}
"""

# --------------------------Gradio Layout-----------------------------
with gr.Blocks(title="Chat RAG", fill_width=True, css=css) as demo:
    gr.Markdown("# Chat RAG: Interactive Coding Assistant"
)
    with gr.Row():
        with gr.Column(scale=7, variant="compact"): #
            chatbot = gr.Chatbot(label="Chat RAG", height="80vh")
            msg = gr.Textbox(placeholder="Enter your message here and hit return when you're ready...",
                             interactive=True, container=False, autoscroll=True)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot],
                                       value="Clear Chat Window",
                                       elem_id="button")
                clear_chat_mem = gr.Button(value="Clear Chat Window and Chat Memory",
                                           elem_id="button")
        with gr.Column(scale=3): #
            with gr.Tab("Chat With Files"):
                files = gr.Files(interactive=True,
                                 file_count="multiple",
                                 file_types=["text", ".pdf", ".xlsx", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                             ".css", ".cpp", ".html", ".docx", ".doc", ".js", ".json", ".csv"])
                with gr.Row():
                    upload = gr.Button(value="Upload Data to Knowledge Base",
                                       interactive=True,
                                       size="sm",
                                       elem_id="button")
                    clear_db = gr.Button(value="Clear Knowledge Base",
                                         interactive=True,
                                         size="sm",
                                         elem_id="button")
            # TODO Finish neo4j implementation
            with gr.Tab("Chat With a Database"):
                db_files = gr.Files(interactive=True,
                                    file_count="multiple",
                                    file_types=["text", ".pdf", ".xlsx", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                                ".css", ".cpp", ".html", ".docx", ".doc", ".js", ".json", ".csv"])
                with gr.Row():
                    neo_un = gr.Textbox(label="Neo4j Database Name",
                                        placeholder="Enter Database Name Here...",
                                        interactive=True)
                    neo_pw = gr.Textbox(label="Neo4j Database Password",
                                        placeholder="Enter Database Password Here...",
                                        interactive=True)
                    neo_url = gr.Textbox(label="Neo4j Database Link",
                                         placeholder="Enter Database Link Here...",
                                         interactive=True)
                with gr.Row():
                    upload_db_files = gr.Button("Upload Data to Database",
                                                interactive=True,
                                                size="sm",
                                                elem_id="button")
                    create_db = gr.Button("Load Database to Model",
                                          interactive=True,
                                          size="sm",
                                          elem_id="button")
                    delete_db = gr.Button("Remove Database from Model",
                                          interactive=True,
                                          size="sm",
                                          elem_id="button")
            with gr.Tab("Chat With a GitHub Repository"):
                repoOwnerUsername = gr.Textbox(label="GitHub Repository Owners Username:",
                                               placeholder="Enter GitHub Repository Owners Username Here....",
                                               interactive= True)
                repoName = gr.Textbox(label="GitHub Repository Name:",
                                      placeholder="Enter Repository Name Here....",
                                      interactive= True)
                repoBranch = gr.Textbox(label="GitHub Repository Branch Name:",
                                        placeholder="Enter Branch Name Here....",
                                        interactive=True)
                with gr.Row():
                    getRepo = gr.Button(value="Load Repository to Model",
                                        size="sm",
                                        interactive=True,
                                        elem_id="button")
                    removeRepo = gr.Button(value="Reset Info and Remove Repository from Model",
                                           size="sm",
                                           interactive=True,
                                           elem_id="button")
            choices = ["Ollama"]
            if "HUGGINGFACE_HUB_TOKEN" in os.environ:
                choices.append("HuggingFace")
            if "NVIDIA_API_KEY" in os.environ:
                choices.append("NVIDIA NIM")
            if "OPENAI_API_KEY" in os.environ:
                choices.append("OpenAI")
            if "ANTHROPIC_API_KEY" in os.environ:
                choices.append("Anthropic")
            model_provider = gr.Radio(label="Select Model Provider",
                                      value="Ollama",
                                      choices=choices,
                                      interactive=True,
                                      info="Choose your model provider.")


            @gr.render(inputs=model_provider)
            def render_provider_components(provider):
                if provider == "Ollama":
                    # --------------Ollama Components------------------------
                    selected_chat_model = gr.Dropdown(choices=list(modelUtils.model_display_names["Ollama"].keys()),
                                                      interactive=True,
                                                      label="Select a Chat Model",
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
                    # ---------Ollama Buttons-----------------
                    selected_chat_model.change(gradioUtils.update_model,
                                               inputs=[selected_chat_model],
                                               outputs=[chatbot])
                    temperature.release(gradioUtils.update_model_temp,
                                        inputs=[temperature])
                    max_tokens.release(gradioUtils.update_max_tokens,
                                       inputs=[max_tokens])
                    custom_prompt.submit(gradioUtils.update_chat_prompt,
                                         inputs=[custom_prompt])

                elif provider == "HuggingFace":
                    # ------------------HuggingFace components-------------------------------
                    hf_quantization = gr.Dropdown(choices=["Choose a Quantization","No Quantization", "2 Bit", "4 Bit", "8 Bit"],
                                                  interactive=True,
                                                  label="Model Quantization",
                                                  value="Choose a Quantization",
                                                  info="Choose Model Quantization.")
                    hf_model = gr.Dropdown(choices=list(modelUtils.model_display_names["HuggingFace"].keys()),
                                           interactive=True,
                                           label="Select a Chat Model",
                                           value="Choose a Model",
                                           filterable=True,
                                           info="Choose a Hugging Face model.")
                    hf_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                                               label="Model Temperature",
                                               info="Select a Temperature between 0 and 1 for the model.",
                                               interactive=True)
                    hf_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                                         label="Top P",
                                         info="Select a Top P value between 0 and 1 for the model.",
                                         interactive=True)
                    hf_ctx_wnd = gr.Slider(minimum=100, maximum=10000, value=2048, step=1,
                                           label="Context Window",
                                           info="Select a Context Window value between 100 and 10000 for the model.",
                                           interactive=True)
                    hf_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                                              label="Max Output Tokens",
                                              info="Set the maximum number of tokens the model can respond with.",
                                              interactive=True)
                    hf_custom_prompt = gr.Textbox(label="Enter a Custom Prompt",
                                                  placeholder="Enter your custom prompt here...",
                                                  interactive=True)
                    # ---------HuggingFace Buttons-----------------
                    hf_model.change(gradioUtils.update_model,
                                    inputs=[hf_model],
                                    outputs=[chatbot])
                    hf_quantization.change(gradioUtils.update_quant,
                                           inputs=[hf_quantization])
                    hf_temperature.release(gradioUtils.update_model_temp,
                                           inputs=[hf_temperature])
                    hf_top_p.release(gradioUtils.update_top_p,
                                     inputs=[hf_top_p])
                    hf_ctx_wnd.release(gradioUtils.update_context_window,
                                       inputs=[hf_ctx_wnd])
                    hf_max_tokens.release(gradioUtils.update_max_tokens,
                                          inputs=[hf_max_tokens])
                    hf_custom_prompt.submit(gradioUtils.update_chat_prompt,
                                            inputs=[hf_custom_prompt])

                elif provider=="NVIDIA NIM":
                    # ----------------------------NVIDIA NIM components---------------------------
                    nv_model = gr.Dropdown(choices=list(modelUtils.model_display_names["NVIDIA NIM"].keys()),
                                           interactive=True,
                                           label="Select a NVIDIA NIM",
                                           value="Codestral 22B",
                                           filterable=True,
                                           info="Choose a NVIDIA NIM.")
                    nv_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                                               label="Model Temperature",
                                               info="Select a temperature between .1 and 1 to set the model to.",
                                               interactive=True)
                    nv_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                                         label="Top P",
                                         info="Set the top p value for the model.",
                                         interactive=True)
                    nv_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                                              label="Max Output Tokens",
                                              info="Set the maximum number of tokens the model can respond with.",
                                              interactive=True)
                    # ---------NVIDIA Buttons-----------------
                    nv_model.change(gradioUtils.update_model,
                                    inputs=[nv_model],
                                    outputs=[chatbot])
                    nv_temperature.release(gradioUtils.update_model_temp,
                                           inputs=[nv_temperature])
                    nv_top_p.release(gradioUtils.update_top_p,
                                     inputs=[nv_top_p])
                    nv_max_tokens.release(gradioUtils.update_max_tokens,
                                          inputs=[nv_max_tokens])

                elif provider=="OpenAI":
                    # ----------------------------OPEN AI components---------------------------
                    openai_model = gr.Dropdown(choices=list(modelUtils.model_display_names["OpenAI"].keys()),
                                               interactive=True,
                                               label="Select a OpenAI Model",
                                               value="GPT-4o",
                                               filterable=True,
                                               info="Choose a OpenAI model.")
                    openai_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                                                   label="Model Temperature",
                                                   info="Select a temperature between .1 and 1 to set the model to.",
                                                   interactive=True)
                    openai_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                                             label="Top P",
                                             info="Set the top p value for the model.",
                                             interactive=True)
                    openai_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                                                  label="Max Output Tokens",
                                                  info="Set the maximum number of tokens the model can respond with.",
                                                  interactive=True)
                    # ---------OpenAI Buttons-----------------
                    openai_model.change(gradioUtils.update_model,
                                        inputs=[openai_model],
                                        outputs=[chatbot])
                    openai_temperature.release(gradioUtils.update_model_temp,
                                               inputs=[openai_temperature])
                    openai_top_p.release(gradioUtils.update_top_p,
                                         inputs=[openai_top_p])
                    openai_max_tokens.release(gradioUtils.update_max_tokens,
                                              inputs=[openai_max_tokens])
                elif provider=="Anthropic":
                    # ----------------------------Anthropic components---------------------------
                    anth_model = gr.Dropdown(choices=list(modelUtils.model_display_names["Anthropic"].keys()),
                                             interactive=True,
                                             label="Select a Anthropic Model",
                                             value="Claude 3.5 Sonnet",
                                             filterable=True,
                                             info="Choose a Anthropic model.")
                    anth_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                                                 label="Model Temperature",
                                                 info="Select a temperature between .1 and 1 to set the model to.",
                                                 interactive=True)
                    anth_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                                                label="Max Output Tokens",
                                                info="Set the maximum number of tokens the model can respond with.",
                                                interactive=True)
                    # ---------Anthropic Buttons-----------------
                    anth_model.change(gradioUtils.update_model,
                                      inputs=[anth_model],
                                      outputs=[chatbot])
                    anth_temperature.release(gradioUtils.update_model_temp,
                                             inputs=[anth_temperature])
                    anth_max_tokens.release(gradioUtils.update_max_tokens,
                                            inputs=[anth_max_tokens])
                gradioUtils.update_model_provider(provider)
# ----------------------------------Button Functionality For RAG Chat-----------------------------------------------
        msg.submit(gradioUtils.stream_response,
                   inputs=[msg],
                   outputs=[msg, chatbot],
                   show_progress="full",
                   scroll_to_output=True)
    # --------------------Buttons in Left Column--------------------------------
        clear.click(gradioUtils.clear_chat_history,
                    outputs=chatbot)
        clear_chat_mem.click(gradioUtils.clear_his_and_mem,
                             outputs=chatbot)
    # --------------------Buttons in Right Column--------------------------------
        files.upload(gradioUtils.handle_doc_upload, inputs=files,
                     show_progress="full")
        upload.click(lambda: gradioUtils.model_manager.reset_chat_engine())
        clear_db.click(gradioUtils.delete_db,
                       show_progress="full")
        getRepo.click(gradioUtils.set_github_info, inputs=[repoOwnerUsername, repoName, repoBranch])
        removeRepo.click(modelUtils.reset_github_info, outputs=[repoOwnerUsername, repoName, repoBranch])

demo.launch(inbrowser=True)
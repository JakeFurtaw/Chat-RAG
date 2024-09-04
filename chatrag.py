import os
import gradio as gr
from gradio_utils import GradioUtils
from model_utils import ModelManager
import dotenv
dotenv.load_dotenv()

gradioUtils = GradioUtils()
modelUtils = ModelManager()
# --------------Ollama Components------------------------
selected_chat_model = gr.Dropdown(choices=list(modelUtils.ollama_model_display_names.keys()),
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

# ------------------HuggingFace components-------------------------------
hf_model = gr.Dropdown(choices=list(modelUtils.hf_model_display_names.keys()),
                       interactive=True,
                       label="Select a Chat Model",
                       value="Codestral 22B",
                       filterable=True,
                       info="Choose a Hugging Face model.",
                       visible=False)
hf_quantization = gr.Dropdown(choices=["No Quantization","2 Bit","4 Bit", "8 Bit"],
                              interactive=True,
                              label="Model Quantization",
                              value="4 Bit",
                              info="Choose Model Quantization.",
                              visible=False)
hf_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                           label="Model Temperature",
                           info="Select a Temperature between 0 and 1 for the model.",
                           interactive=True,
                           visible=False)
hf_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                     label="Top P",
                     info="Select a Top P value between 0 and 1 for the model.",
                     interactive=True,
                     visible=False)
hf_ctx_wnd = gr.Slider(minimum=100, maximum=10000, value=2048, step=1,
                     label="Context Window",
                     info="Select a Context Window value between 100 and 10000 for the model.",
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
nv_model = gr.Dropdown(choices=list(modelUtils.nv_model_display_names.keys()),
                        interactive=True,
                        label="Select a NVIDIA NIM",
                        value="Codestral 22B",
                        filterable=True,
                        info="Choose a NVIDIA NIM.",
                        visible=False)
nv_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                           label="Model Temperature",
                           info="Select a temperature between .1 and 1 to set the model to.",
                           interactive=True,
                           visible=False)
nv_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                     label="Top P",
                     info="Set the top p value for the model.",
                     interactive=True,
                     visible=False)
nv_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                          label="Max Output Tokens",
                          info="Set the maximum number of tokens the model can respond with.",
                          interactive=True,
                          visible=False)

# ----------------------------OPEN AI components---------------------------
openai_model = gr.Dropdown(choices=list(modelUtils.openai_model_display_names.keys()),
                           interactive=True,
                           label="Select a OpenAI Model",
                           value="GPT-4o",
                           filterable=True,
                           info="Choose a OpenAI model.",
                           visible=False)
openai_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                               label="Model Temperature",
                               info="Select a temperature between .1 and 1 to set the model to.",
                               interactive=True,
                               visible=False)
openai_top_p = gr.Slider(minimum=0, maximum=1, value=0.4, step=0.05,
                         label="Top P",
                         info="Set the top p value for the model.",
                         interactive=True,
                         visible=False)
openai_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                              label="Max Output Tokens",
                              info="Set the maximum number of tokens the model can respond with.",
                              interactive=True,
                              visible=False)

# ----------------------------Anthropic components---------------------------
anth_model = gr.Dropdown(choices=list(modelUtils.anth_model_display_names.keys()),
                         interactive=True,
                         label="Select a Anthropic Model",
                         value="Claude 3.5 Sonnet",
                         filterable=True,
                         info="Choose a Anthropic model.",
                         visible=False)
anth_temperature = gr.Slider(minimum=0, maximum=1, value=.75, step=.05,
                             label="Model Temperature",
                             info="Select a temperature between .1 and 1 to set the model to.",
                             interactive=True,
                             visible=False)
anth_max_tokens = gr.Slider(minimum=100, maximum=5000, value=2048, step=1,
                            label="Max Output Tokens",
                            info="Set the maximum number of tokens the model can respond with.",
                            interactive=True,
                            visible=False)

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
                                 label="Upload Files Here",
                                 file_count="multiple",
                                 file_types=["text", ".pdf", ".py", ".txt", ".dart", ".c", ".jsx", ".xml",
                                             ".css", ".cpp", ".html", ".docx", ".doc", ".js", ".json"])
                with gr.Row():
                    upload = gr.Button(value="Upload Data",
                                       interactive=True,
                                       size="sm",
                                       elem_id="button")
                    clear_db = gr.Button(value="Clear RAG Database",
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
                    removeRepo = gr.Button(value="Reset Info",
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
            selected_chat_model.render()
            temperature.render()
            max_tokens.render()
            custom_prompt.render()

            hf_model.render()
            hf_quantization.render()
            hf_temperature.render()
            hf_top_p.render()
            hf_ctx_wnd.render()
            hf_max_tokens.render()
            hf_custom_prompt.render()

            nv_model.render()
            nv_temperature.render()
            nv_top_p.render()
            nv_max_tokens.render()

            openai_model.render()
            openai_temperature.render()
            openai_top_p.render()
            openai_max_tokens.render()

            anth_model.render()
            anth_temperature.render()
            anth_max_tokens.render()

            def update_layout(choice):
                ollama_visible = choice == "Ollama"
                hf_visible = choice == "HuggingFace"
                nv_visible = choice == "NVIDIA NIM"
                oai_visible = choice == "OpenAI"
                ath_visible = choice == "Anthropic"
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
                    gr.update(visible=hf_visible),
                    gr.update(visible=hf_visible),

                    gr.update(visible=nv_visible),
                    gr.update(visible=nv_visible),
                    gr.update(visible=nv_visible),
                    gr.update(visible=nv_visible),

                    gr.update(visible=oai_visible),
                    gr.update(visible=oai_visible),
                    gr.update(visible=oai_visible),
                    gr.update(visible=oai_visible),

                    gr.update(visible=ath_visible),
                    gr.update(visible=ath_visible),
                    gr.update(visible=ath_visible),
                )

            def update_model_options(choice):
                if choice == "Ollama":
                    return gr.update(choices=list(modelUtils.ollama_model_display_names.keys()),
                                     value="Codestral 22B")
                elif choice == "HuggingFace":
                    return gr.update(choices=list(modelUtils.hf_model_display_names.keys()),
                                     value="Codestral 22B")
                elif choice == "NVIDIA NIM":
                    return gr.update(choices=list(modelUtils.nv_model_display_names.keys()),
                                     value="Codestral 22B")
                elif choice == "OpenAI":
                    return gr.update(choices=list(modelUtils.openai_model_display_names.keys()),
                                     value="GPT-4o")
                elif choice == "Anthropic":
                    return gr.update(choices=list(modelUtils.anth_model_display_names.keys()),
                                     value="Claude 3.5 Sonnet")
                else:
                    return ValueError(f"{choice} not supported")

            def update_layout_and_model(choice):
                layout_updates = update_layout(choice)
                model_options_update = update_model_options(choice)
                gradioUtils.update_model_provider(choice)
                return layout_updates + (model_options_update,)

            model_provider.change(
                fn=update_layout_and_model,
                inputs=[model_provider],
                outputs=[
                    selected_chat_model, temperature, max_tokens, custom_prompt,
                    hf_model, hf_quantization, hf_temperature, hf_top_p, hf_ctx_wnd, hf_max_tokens, hf_custom_prompt,
                    nv_model, nv_temperature, nv_top_p, nv_max_tokens,
                    openai_model, openai_temperature, openai_top_p, openai_max_tokens,
                    anth_model, anth_temperature, anth_max_tokens
                ]
            )
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
        files.upload(gradioUtils.handle_doc_upload,
                     show_progress="full")
        upload.click(lambda: gradioUtils.model_manager.reset_chat_engine())
        clear_db.click(gradioUtils.delete_db,
                       show_progress="full")
        getRepo.click(gradioUtils.set_github_info, inputs=[repoOwnerUsername, repoName, repoBranch])
        removeRepo.click(modelUtils.reset_github_info, outputs=[repoOwnerUsername, repoName, repoBranch])
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
        # ---------Anthropic Buttons-----------------
        anth_model.change(gradioUtils.update_model,
                          inputs=[anth_model],
                          outputs=[chatbot])
        anth_temperature.release(gradioUtils.update_model_temp,
                                 inputs=[anth_temperature])
        anth_max_tokens.release(gradioUtils.update_max_tokens,
                                inputs=[anth_max_tokens])

demo.launch(inbrowser=True, share=True)
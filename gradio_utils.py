import os, shutil
import gradio as gr
from model_utils import ModelManager, ModelParamUpdates

"""
Main gradio class that is the connector function between the front and backend. This function serves many purposes
from updating model parameters and calling the appropriate function to handing the response streaming after calling the 
process input function.
This class also handles chat memory, deleting the knowledge base, and handing the document uploads from the main gradio 
file asset.  
"""
class GradioUtils:
    def __init__(self):
        self.model_manager = ModelManager()
        self.model_param_updater = ModelParamUpdates(self.model_manager)
        self.chat_history = []

    # This function gets the users query, send it to the chat engine to be processed and then streams the response back
    def stream_response(self, message: str):
        streaming_response = self.model_manager.process_input(message)
        full_response = ""
        for tokens in streaming_response.response_gen:
            full_response += tokens
            yield "", self.chat_history + [(message, full_response)]
        self.chat_history.append((message, full_response))

    # This function clears the chat history dictionary
    def clear_chat_history(self):
        self.chat_history.clear()

    """
    This function clears the chat history and reset the chat engine to clear the model memory to give the user a 
    fresh chat with no context
    """
    def clear_his_and_mem(self):
        self.clear_chat_history()
        self.model_manager.reset_chat_engine()

    """
    This function deletes the data file to remove the uploaded data and resets the chat engine to remove it from the
    models' context. It also sends the warning message to the front end to alert the user of the changes made.
    """
    def delete_db(self):
        gr.Info("Wait about 10 seconds for the files to clear. After this message disappears you should  "
                "be in the clear.", duration=15)
        if os.path.exists("data"):
            shutil.rmtree("data")
            os.makedirs("data")
        self.model_manager.reset_chat_engine()

    """
    This function clears the chat history, updates the model provider based off users selection and then sends a
    #warning message about model loading and downloading wait times if the user requested to use a huggingface model.
    """
    def update_model_provider(self, provider):
        self.clear_chat_history()
        self.model_manager.update_model_provider(provider)
        if self.model_manager.provider == "HuggingFace":
            gr.Warning(
                "If this is your first time using HuggingFace the model may need to download. Please be patient.",
                duration=10)

    # This function sends the users model selection through to the model manager
    def update_model(self, display_name):
        self.clear_chat_history()
        self.model_manager.update_model(display_name)

    # This function sends the users quantization selection through to the model parameter updater function
    def update_quant(self, quantization):
        self.model_param_updater.update_quant(quantization)

    # This function sends the users quantization selection through to the model parameter updater function
    def update_model_temp(self, temperature):
        self.model_param_updater.update_model_temp(temperature)

    # This function sends the users top p selection through to the model parameter updater function
    def update_top_p(self, top_p):
        self.model_param_updater.update_top_p(top_p)

    # This function sends the users context window size selection through to the model parameter updater function
    def update_context_window(self, context_window):
        self.model_param_updater.update_context_window(context_window)

    # This function sends the users max token selection through to the model parameter updater function
    def update_max_tokens(self, max_tokens):
        self.model_param_updater.update_max_tokens(max_tokens)

    # This function sends the users custom prompt through to the model parameter updater function
    def update_chat_prompt(self, custom_prompt):
        self.model_param_updater.update_chat_prompt(custom_prompt)

    # This function sends the users GitHub repo info through to the model manager function
    def set_github_info(self, owner, repo, branch):
        self.model_manager.set_github_info(owner, repo, branch)

    """
    This function handles the document uploading and sending the user a message about what to do for the model 
    to see the files.
    """
    @staticmethod
    def handle_doc_upload(files):
        gr.Warning("Make sure you hit the upload button or the model wont see your files!", duration=10)
        return [file.name for file in files]

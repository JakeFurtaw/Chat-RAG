# Chat RAG: Interactive Coding Assistant

## Overview

CodeChat RAG is an interactive coding assistant that leverages Retrieval-Augmented Generation (RAG) to provide 
informed responses to coding queries. Built with a user-friendly Gradio interface, it allows users to interact with 
various language models and upload context files for more accurate assistance.

## Features

- **Multiple Language Models**: Choose from models like Codestral, Mistral-Nemo, LLaMA3.1, DeepSeek Coder v2, Gemma2, and CodeGemma.
- **RAG-powered Responses**: Utilizes uploaded documents to provide context-aware answers.
- **Interactive Chat Interface**: Easy-to-use chat interface for asking coding questions.
- **File Upload**: Support for uploading additional context files.
- **Model Switching**: Seamlessly switch between different language models.
- **Reset Chat Engine**: Clear chat history and memory to start fresh.
- **Delete Database**: Easily delete all stored data for privacy and reset purposes.
- **Enhanced Memory Management**: Dynamically manage chat memory for different models.
- **Refined Chat Prompts**: Contextual prompts guide the AI for more accurate and useful responses.


## Usage

1. Run the application:
    ```bash
    python app.py
   ```
2. Open the provided URL in your web browser.
3. (Optional) Upload relevant files for additional context.
4. Select a language model from the dropdown menu.
5. Type your coding question in the text box and press enter.


## Project Structure

- `app.py`: Main application entry point
- `chat.py`: Core chat functionality, including document loading and chat engine setup
- `cr_gradio.py`: Gradio interface setup and management
- `model_utils.py`: Manages model selection, memory, and user input processing
- `utils.py`: Utility functions for embedding, LLM setup, and chat engine configuration

## Pictures
### Start State of the App
![Start State of the App](pics/start_state.png "Start State of the App")
### Dropdown Menu in Action
![Dropdown Menu](pics/model_dropdown.png "Dropdown Menu in Action")
### RAG Query Example
![RAG Query Example](pics/RAG_Query.png "RAG Query Example")


### Need Help or Have Feature Suggestions
Feel free to reach out to me through GitHub, LinkedIn, or through email. All of those are avialable on my website [JFCoded](https://www.jfcoded.com/contact).
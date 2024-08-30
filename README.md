# Chat RAG: Interactive Coding Assistant

## Overview

Chat RAG is an advanced interactive coding assistant that leverages Retrieval-Augmented Generation (RAG) to provide 
informed responses to coding queries. Built with a user-friendly Gradio interface, it allows users to interact 
with various language models, customize model parameters, and upload context files for more accurate assistance.

## Features

- **Multiple Model Providers**: Support for Ollama, HuggingFace, and NVIDIA NIM models.
- **Wide Range of Language Models**: Choose from models like Codestral, Mistral-Nemo, LLaMA3.1, DeepSeek Coder v2,
Gemma2, and CodeGemma.
- **RAG-powered Responses**: Utilizes uploaded documents to provide context-aware answers.
- **Interactive Chat Interface**: Easy-to-use chat interface for asking coding questions.
- **File Upload**: Support for uploading additional context files.
- **Model Switching**: Seamlessly switch between different language models.
- **Customizable Model Parameters**: Adjust temperature, max tokens, top-p, and context window size.
- **Custom Prompts**: Ability to set custom system prompts for the chat engine.
- **Reset Chat Engine**: Clear chat history and memory to start fresh.
- **Delete Database**: Easily delete all stored data for privacy and reset purposes.
- **Enhanced Memory Management**: Dynamically manage chat memory for different models.
- **Streaming Responses**: Real-time response generation for a more interactive experience.
- **Model Quantization**: Options for 2-bit, 4-bit, and 8-bit quantization for HuggingFace models.


## Setup and Usage

1. Clone the repository.
2. Install the required dependencies.
3. Set up your .env file with the following:
   ```bash
   GRADIO_TEMP_DIR="YourPathTo/Chat-RAG/data"
   GRADIO_WATCH_DIRS="YourPathTo/Chat-RAG"
   HUGGINGFACE_HUB_TOKEN="YOUR HF TOKEN HERE"
   ```
4. Run the application:
    ```bash
    gradio chatrag.py
   ```
5. The app will automatically open a new tab and launch in your browser.
6. (Optional) Upload relevant files for additional context.
7. Select a Model Provider.
8. Select a language model from the dropdown menu.
9. Type your coding question in the text box and press enter.
10. The model will stream the response to your query back to you in the chat window.


## Project Structure

- `chatrag.py`: Main application file with Gradio UI setup.
- `chat.py`: Utilities for document loading and chat engine creation.
- `gr_utils.py`: Gradio-specific utility functions for UI interactions.
- `model_utils.py`: Model management and configuration utilities.
- `utils.py`: General utilities for embedding, LLM setup, and chat memory.

## Pictures
### Start State of the App
![Start State of the App](pics/start_state.png "Start State of the App")
### Dropdown Menu in Action
![Dropdown Menu](pics/model_dropdown.png "Dropdown Menu in Action")
### Query Example
![Query Example](pics/query.png "Query Example")
### RAG Query Example
![RAG Query Example](pics/RAG_Query.png "RAG Query Example")

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request or Fork the Repository.

### Need Help or Have Feature Suggestions?
Feel free to reach out to me through GitHub, LinkedIn, or through email. All of those are available on my website [JFCoded](https://www.jfcoded.com/contact).
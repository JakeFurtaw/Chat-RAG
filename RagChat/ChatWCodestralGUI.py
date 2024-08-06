import gradio as gr
import queue
import threading
from ChatWCodestral import main as CWCMain
import dotenv


class CWCGradio:
    dotenv.load_dotenv()

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()

        self.CWC_thread = threading.Thread(target=self.run_cwc)
        self.CWC_thread.start()

    def run_cwc(self):
        def send_input():
            msg = self.input_queue.get()
            return msg

        def input_print(message):
            self.output_queue.put(message)

        CWCMain(send_input=send_input, input_print=input_print)

    def chat(self, message, history):
        self.input_queue.put(message)
        try:
            response = self.output_queue.get(timeout=60)
            history.append((message, response))
            return "", history  # Return an empty string for the message input and the updated history
        except queue.Empty:
            error_message = "I'm sorry, I'm having trouble responding right now. Please try again."
            history.append((message, error_message))
            return "", history

    def launch(self):
        with gr.Blocks(theme="monochrome", fill_height=True, fill_width=True) as iface:
            gr.Markdown("# Chat With Codestral using RAG")
            gr.Markdown("Input your coding question and let the model do the rest! You can also upload files to give"
                        " the model context to better answer your question with.")
            with gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(autoscroll=True, autofocus=True, placeholder="Enter your coding question here...")
                    with gr.Row():
                        clear = gr.ClearButton([msg, chatbot])
                    msg.submit(self.chat, inputs=[msg, chatbot], outputs=[msg, chatbot])
                with gr.Column(scale=1):
                    gr.Files()

        iface.launch(inbrowser=True, share=True)


if __name__ == "__main__":
    app = CWCGradio()
    app.launch()
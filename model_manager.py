import queue
import threading
from chat import main as CWCMain
import dotenv

dotenv.load_dotenv()


class ModelManager:

    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.selected_model = "codestral:latest"
        self.stop_thread = threading.Event()
        self.thread = threading.Thread(target=self.run_model)
        self.thread.start()

    def run_model(self):
        def send_input():
            msg = self.input_queue.get()
            if msg == "__stop__":
                return None
            return msg

        def input_print(message):
            self.output_queue.put(message)

        while not self.stop_thread.is_set():
            try:
                CWCMain(model=self.selected_model, send_input=send_input, input_print=input_print)
            except Exception as e:
                input_print(f"Error: {str(e)}")
                break

    def process_input(self, message):
        self.input_queue.put(message)
        try:
            return self.output_queue.get(timeout=60)
        except queue.Empty:
            return "I'm sorry, I'm having trouble responding right now. Please try again."

    def update_model(self, model):
        self.selected_model = model
        self.stop_thread.set()
        self.input_queue.put("__stop__")
        self.thread.join()

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stop_thread.clear()
        self.thread = threading.Thread(target=self.run_model)
        self.thread.start()

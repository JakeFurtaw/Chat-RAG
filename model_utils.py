import queue
import threading
from chat import create_chat_engine
import dotenv

dotenv.load_dotenv()


class ModelManager:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.selected_model = "codestral:latest"
        self.model_temp = .75
        self.stop_thread = threading.Event()
        self.chat_engine = create_chat_engine(self.selected_model, self.model_temp)
        self.thread = threading.Thread(target=self.run_model)
        self.thread.start()

    def run_model(self):
        while not self.stop_thread.is_set():
            try:
                query = self.input_queue.get()
                if query == "__stop__":
                    break
                response = self.chat_engine.chat(query)
                self.output_queue.put(str(response))
            except Exception as e:
                self.output_queue.put(f"Error: {str(e)}")
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
        self.chat_engine = create_chat_engine(self.selected_model, self.model_temp)
        self.thread = threading.Thread(target=self.run_model)
        self.thread.start()

    def reset_chat_engine(self):
        self.chat_engine = create_chat_engine(self.selected_model, self.model_temp)

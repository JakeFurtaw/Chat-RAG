from llama_index.core.llms import ChatMessage
from utils import set_llm


def main() -> None:
    llm = set_llm()
    while (query := input("Enter your coding question (e to exit): ")) != "e":
        question = [ChatMessage(content=query)]
        response_generator = llm.stream_chat(question)
        # Print the streamed response
        for response in response_generator:
            print(response.delta, end='', flush=True)
        print("\n")  # Add a newline after the full response


if __name__ == "__main__":
    main()

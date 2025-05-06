"""
Standard config file that stores lists so they don't take up room in the main files.
"""

OLLAMA_MODEL_LIST = {
            "Codestral": "codestral:latest",
            "Qwen 3": "qwen3:latest",
            "Gemma 3": "gemma3:12b",
            "CodeGemma": "codegemma:latest",
            "Mistral-Nemo": "mistral-nemo:latest",
            "Llama3.1": "llama3.1:latest",
            "Deepseek R-1": "deepseek-r1:14b",
            "DeepSeek Coder V2": "deepseek-coder-v2:latest"
        }
HF_MODEL_LIST = {
            "Choose a Model": "",
            "Codestral 22B": "mistralai/Codestral-22B-v0.1",
            "Qwen 3": "Qwen/Qwen3-14B",
            "Gemma 3":"google/gemma-3-12b-it",
            "CodeGemma 7B-Instruct": "google/codegemma-7b-it",
            "Mistral-Nemo 12B-Instruct": "mistralai/Mistral-Nemo-Instruct-2407",
            "Llama3.1 8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "DeepSeek Coder V2 16B": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",

        }
NV_MODEL_LIST = {
            "Codestral 22B": "mistralai/codestral-22b-instruct-v0.1",
            "Qwen 32B": "qwen/qwq-32b",
            "Gemma 3": "google/gemma-3-27b-it",
            "CodeGemma 7B": "google/codegemma-7b",
            "Mistral-Nemo 12B": "nv-mistralai/mistral-nemo-12b-instruct",
            "Llama 3.1 8B": "meta/llama-3.1-8b-instruct",

        }
OA_MODEL_LIST = {"GPT-4o": "gpt-4o",
                 "GPT-4o mini": "gpt-4o-mini",
                 "GPT-4": "gpt-4",
        }
ANTH_MODEL_LIST = {"Claude 3.7 Sonnet": "claude-3-7-sonnet-20250219",
                   "Claude 3.5 Sonnet": "claude-3-5-sonnet-20241022",
                   "Claude 3.5 Haiku": "claude-3-5-haiku-20241022",
                   "Claude 3 Opus": "claude-3-opus-20240229",
                   "Claude 3 Sonnet": "claude-3-sonnet-20240229",
        }

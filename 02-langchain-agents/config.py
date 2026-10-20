"""
config.py: model setup for the LangChain lessons.

In folder 01 we built an `OpenAI(base_url=..., api_key=...)` client by hand.
LangChain replaces that with ONE line:

    init_chat_model("groq:openai/gpt-oss-120b")
                     ^^^^ ^^^^^^^^^^^^^^^^^^^
                provider      model name

Switching provider = changing that string (and installing the provider package):
    "groq:openai/gpt-oss-120b"          pip install langchain-groq
    "google_genai:gemini-2.5-flash"     pip install langchain-google-genai
    "ollama:qwen2.5:7b"                 pip install langchain-ollama
    "anthropic:claude-sonnet-5-5"       pip install langchain-anthropic
    "openai:gpt-5"                      pip install langchain-openai
"""

import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # finds the .env in the parent folder automatically

if not os.getenv("GROQ_API_KEY"):
    sys.exit("Missing GROQ_API_KEY. Put it in the .env file in the project root.")

MODEL_NAME = os.getenv("MODEL", "openai/gpt-oss-120b")
MODEL_ID = f"groq:{MODEL_NAME}"  # "provider:model", the format create_agent() also accepts


def get_model(**kwargs):
    """Return a LangChain chat model. kwargs: temperature, max_tokens, ..."""
    return init_chat_model(MODEL_ID, **kwargs)


def title(text: str) -> None:
    print(f"\n\033[1;36m=== {text} ===\033[0m")


def print_messages(messages) -> None:
    """Show an agent's message history in a readable way."""
    for m in messages:
        kind = m.type.upper()  # human / ai / tool / system
        if m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                print(f"  \033[33m[AI → tool call]\033[0m {tc['name']}({tc['args']})")
        elif m.type == "tool":
            print(f"  \033[35m[TOOL {m.name}]\033[0m {m.content}")
        else:
            print(f"  \033[32m[{kind}]\033[0m {m.content}")

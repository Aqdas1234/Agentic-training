"""
config.py: the ONE place that knows which LLM provider we talk to.

Groq, Gemini, Ollama, OpenRouter and others all speak the OpenAI
"Chat Completions" format, so we can use the official `openai` Python SDK
with any of them. Switching provider = change BASE_URL, API_KEY and MODEL below.
"""

import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads variables from the .env file into os.environ

# ---- Provider settings (Groq free tier) ----------------------------------
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")

# ---- Other free options (uncomment one block to switch) ------------------
# Google Gemini (key from https://aistudio.google.com/apikey):
# BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# API_KEY = os.getenv("GEMINI_API_KEY")
# MODEL = "gemini-2.5-flash"
#
# Ollama, running locally (`brew install ollama && ollama pull qwen2.5:7b`):
# BASE_URL = "http://localhost:11434/v1"
# API_KEY = "ollama"            # any non-empty string works
# MODEL = "qwen2.5:7b"
#
# OpenRouter free models (key from https://openrouter.ai/keys):
# BASE_URL = "https://openrouter.ai/api/v1"
# API_KEY = os.getenv("OPENROUTER_API_KEY")
# MODEL = "meta-llama/llama-3.3-70b-instruct:free"

if not API_KEY:
    sys.exit("Missing API key. Copy .env.example to .env and paste your GROQ_API_KEY.")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def assistant_to_dict(msg) -> dict:
    """
    Convert the SDK's assistant message into a plain dict we can append to
    `messages`. We copy only the standard fields: some models add extras
    (e.g. `reasoning`) that the API rejects if we send them back.
    """
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


# ---- Small printing helpers so lessons can SHOW what's happening ----------
def show(title: str, data) -> None:
    """Pretty-print anything (dicts, lists, SDK objects) under a heading."""
    print(f"\n\033[1;36m=== {title} ===\033[0m")
    if hasattr(data, "model_dump"):  # SDK response objects -> plain dicts
        data = data.model_dump(exclude_none=True)
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)

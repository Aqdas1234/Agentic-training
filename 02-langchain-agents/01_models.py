"""
LESSON 1: Model integrations. One interface for every LLM provider.

The problem LangChain solves here: every provider (OpenAI, Anthropic, Google,
Groq, Ollama...) has a slightly different SDK and response format.
LangChain wraps them all in the SAME "chat model" interface:

    model.invoke(messages)   -> one AIMessage
    model.stream(messages)   -> chunks as they are generated
    model.batch([m1, m2])    -> several requests in parallel

Messages become typed objects instead of raw dicts:
    SystemMessage  ~ {"role": "system", ...}
    HumanMessage   ~ {"role": "user", ...}
    AIMessage      ~ {"role": "assistant", ...}
    ToolMessage    ~ {"role": "tool", ...}
(Plain dicts like {"role": "user", "content": "hi"} are accepted too.)

Run:  python 01_models.py
"""

from langchain.messages import HumanMessage, SystemMessage

from config import MODEL_ID, get_model, title

model = get_model(temperature=0.3)
print(f"Using model: {MODEL_ID}  ->  Python class: {type(model).__name__}")

# ---- 1. invoke: send messages, get ONE AIMessage back ----------------------
messages = [
    SystemMessage("You are a concise tutor. Answer in one sentence."),
    HumanMessage("What is LangChain?"),
]
response = model.invoke(messages)

title("1. invoke() returns an AIMessage")
print("type:              ", type(response).__name__)
print("content:           ", response.content)
title("usage_metadata (standardized token counts, same keys for every provider)")
print(response.usage_metadata)
title("response_metadata (provider-specific raw info, e.g. finish_reason)")
print({k: response.response_metadata.get(k) for k in ("model_name", "finish_reason")})

# ---- 2. A plain string is shorthand for a single HumanMessage ---------------
title("2. invoke() with a plain string")
print(model.invoke("Say hello in Japanese, one word only.").content)

# ---- 3. stream: tokens arrive as they're generated (what chat UIs use) -------
title("3. stream()")
for chunk in model.stream("Count from 1 to 10, comma separated."):
    print(chunk.content, end="", flush=True)


# ---- 4. batch: several independent requests in parallel ---------------------
title("4. batch()")
answers = model.batch(["Capital of France? One word.", "Capital of Japan? One word."])
for a in answers:
    print("-", a.content)

# ---- Compare with folder 01 --------------------------------------------------
# Before: client.chat.completions.create(model=..., messages=[...]).choices[0].message.content
# Now:    model.invoke([...]).content
# And to switch to Gemini you'd only change "groq:..." to "google_genai:gemini-2.5-flash".

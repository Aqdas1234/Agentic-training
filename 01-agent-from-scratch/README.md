# Agentic Training: build an LLM agent from scratch

A hands-on course in 5 small Python files. By the end you'll understand:
**messages → model responses → tool calling → the agent loop.**

Uses the `openai` Python SDK with **Groq's free tier** (any OpenAI-compatible provider works).

---

## 1. Setup (5 minutes)

```bash
cd "Agentic training"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your free key from https://console.groq.com/keys

cd 01-agent-from-scratch        # run the lessons from inside this folder
```

`.venv`, `.env` and `requirements.txt` live in the project root and are shared with `02-langchain-agents`.

Test the tools without any API key:
```bash
python tools.py
```

Then run the lessons in order:
```bash
python 01_first_message.py
python 02_conversation.py
python 03_single_tool_call.py
python 04_agent_loop.py
python 05_chat_agent.py
```

---

## 2. Free LLM options

| Provider | Free? | Tool calling | Notes |
|---|---|---|---|
| **Groq** (default) | Free key, no card | Good (gpt-oss-120b, gpt-oss-20b, qwen3) | Extremely fast; rate-limited per minute/day |
| **Google Gemini** | Free key at aistudio.google.com | Very good | Generous free tier; OpenAI-compatible endpoint |
| **Ollama** (local) | 100% free, offline | OK (qwen2.5, llama3.1) | Runs on your Mac; ~5 GB download; slower |
| **OpenRouter** `:free` models | Free key | Varies | Many models, tight limits, list changes often |

To switch, edit the 3 lines (`BASE_URL`, `API_KEY`, `MODEL`) in [config.py](config.py). Ready-made blocks for each provider are already in that file. **No other code changes needed.**

---

## 3. Core concepts

### 3.1 Messages: the only thing the model sees
The model is **stateless**. Every call you send a list of messages:

```python
[
  {"role": "system",    "content": "You are a helpful tutor."},     # rules / personality
  {"role": "user",      "content": "My name is Iris."},             # the human
  {"role": "assistant", "content": "Nice to meet you, Iris!"},      # model's earlier reply
  {"role": "user",      "content": "What's my name?"},
]
```
"Memory" = you keep appending to this list and re-send it. (Lesson 2)

### 3.2 The model response
```python
response.choices[0].message.content      # the text
response.choices[0].message.tool_calls   # tool requests (or None)
response.choices[0].finish_reason        # "stop" | "tool_calls" | "length"
response.usage                           # prompt / completion tokens
```
(Lesson 1)

### 3.3 Tool calling
You describe tools with **JSON Schema**. The model reads the `description` to decide when to use them:

```python
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",   # <- the model decides based on this
    "parameters": {
      "type": "object",
      "properties": {"city": {"type": "string"}},
      "required": ["city"]
    }
  }
}
```

**The model never runs code.** It replies with a *request*:
```json
"tool_calls": [{
  "id": "call_abc123",
  "type": "function",
  "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}
}]
```
Your code runs the function and sends back the result as a `tool` message:
```python
{"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp_c\": 18, ...}"}
```
The `tool_call_id` must match, because that's how the model knows which result belongs to which request. (Lesson 3)

### 3.4 The agent loop
```
            ┌──────────────────────────────────────────────┐
            │                                              │
 user ──▶ messages ──▶  LLM  ──▶ tool_calls? ──yes──▶ run tools in Python
                         │                            append results
                         no                           (role="tool")
                         ▼                                 │
                   final answer  ◀─────────────────────────┘ (loop again)
```
```python
while step < MAX_STEPS:
    msg = llm(messages, tools)
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content                 # done!
    for call in msg.tool_calls:
        result = run_tool(call.function.name, call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
```
That's it. **An agent = an LLM + tools + a loop.** Frameworks like LangChain, the OpenAI Agents SDK and the Claude Agent SDK are wrappers around this loop. (Lesson 4)

---

## 4. Common pitfalls

| Mistake | What happens |
|---|---|
| Not appending the assistant message *before* the tool results | API error: "tool message must follow a tool_calls message" |
| Wrong / missing `tool_call_id` | API error, or the model gets confused |
| No `MAX_STEPS` | Model loops forever, burning your free quota |
| Tool raises an exception and crashes your program | Instead, return the error as text: the model can fix its call (see `run_tool`) |
| Forgetting `json.loads(arguments)` | `arguments` is a **string**, not a dict |
| Vague tool descriptions | Model doesn't call the tool, or calls the wrong one |
| Hitting `429 rate limit` | Free tier limit. Wait a minute, or switch to a smaller model |

---

## 5. Files

| File | What it teaches |
|---|---|
| [config.py](config.py) | Client setup, provider switching, print helpers |
| [tools.py](tools.py) | Tool functions + JSON schemas + safe `run_tool` dispatcher |
| [01_first_message.py](01_first_message.py) | One request, inspect the full response |
| [02_conversation.py](02_conversation.py) | Roles, memory via the message list, token growth |
| [03_single_tool_call.py](03_single_tool_call.py) | One tool round trip, printed step by step |
| [04_agent_loop.py](04_agent_loop.py) | The full agent loop with multiple/parallel tools |
| [05_chat_agent.py](05_chat_agent.py) | Interactive terminal agent (`/verbose`, `/history`) |

---

## 6. Exercises (do these to really learn it)

1. **Add a tool**: e.g. `convert_currency(amount, from, to)` or `write_file(path, text)`. Add the function, its schema, and a registry entry in `tools.py`.
2. **Break it on purpose**: in `04_agent_loop.py`, comment out `messages.append(assistant_to_dict(msg))` and read the error.
3. **Force a tool**: set `tool_choice="required"`, then `"none"`. What changes?
4. **Token tracker**: sum `response.usage.total_tokens` across steps and print the total.
5. **Switch provider**: get a Gemini key, flip the block in `config.py`, and run lesson 4 again.
6. **Real weather**: replace the mock with the free [Open-Meteo API](https://open-meteo.com/) (no key needed).
7. **Next step**: compare your loop with the Claude Agent SDK or OpenAI Agents SDK and spot the same loop inside.

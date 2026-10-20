# 02 · LangChain Agents

In `01-agent-from-scratch` you built an agent by hand. Here you'll rebuild it with **LangChain v1** and see what the framework does for you.

Topics: **model integrations · tools · the agent loop (`create_agent`) · middleware · structured output**

## Run

From the project root (one shared `.venv` and `.env` for both folders):
```bash
source .venv/bin/activate
pip install -r requirements.txt        # adds langchain + langchain-groq
cd 02-langchain-agents
python tools.py                        # see the schemas @tool generates (no API call)
python 01_models.py
python 02_tools.py
python 03_agent.py
python 04_middleware.py
python 05_structured_output.py
```

---

## The big picture: what LangChain replaces

| What you wrote by hand (folder 01) | LangChain | Lesson |
|---|---|---|
| `OpenAI(base_url=..., api_key=...)` | `init_chat_model("groq:openai/gpt-oss-120b")` | 1 |
| `{"role": "user", "content": ...}` dicts | `HumanMessage`, `AIMessage`, `ToolMessage` (dicts still work) | 1 |
| `response.choices[0].message.content` | `model.invoke(msgs).content` | 1 |
| Hand-written JSON schema per tool | `@tool` reads name + docstring + type hints | 2 |
| `json.loads(call.function.arguments)` | `tool_call["args"]` is already a dict | 2 |
| `TOOL_REGISTRY` + `run_tool()` | `tool.invoke(tool_call)` → ready `ToolMessage` | 2 |
| `while` loop in `run_agent()` | `create_agent(model, tools)` | 3 |
| Your own `messages` list for memory | `checkpointer` + `thread_id` | 3 |
| `MAX_STEPS` | `ModelCallLimitMiddleware` | 4 |
| `try/except` in `run_tool()` | `@wrap_tool_call` middleware | 4 |
| *(nothing)* | Typed Pydantic results via `response_format` | 5 |

**The core loop is the same.** LangChain packages it, adds standard hooks, and lets you swap providers.

---

## 1. Model integrations ([01_models.py](01_models.py))

Every provider gets the **same interface**:

```python
model = init_chat_model("groq:openai/gpt-oss-120b", temperature=0.3)
model.invoke(messages)     # → AIMessage
model.stream(messages)     # → chunks, token by token
model.batch([m1, m2])      # → parallel requests
```

To switch provider, change the string (and `pip install langchain-<provider>`):
`"google_genai:gemini-2.5-flash"`, `"ollama:qwen2.5:7b"`, `"anthropic:claude-sonnet-5-5"`, …

An `AIMessage` has:
- `.content`: the text
- `.tool_calls`: tool requests (a normalized list)
- `.usage_metadata`: `input_tokens` / `output_tokens`, **with the same keys for every provider**
- `.response_metadata`: the provider's raw extras (`finish_reason`, etc.)

## 2. Tools ([tools.py](tools.py), [02_tools.py](02_tools.py))

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""     # ← the model reads this!
    ...
```
- **Name** = function name, **description** = docstring, **schema** = type hints.
- `Annotated[str, "description"]` describes an individual argument.
- `model.bind_tools(tools)` returns a model that can request them.
- `tool.invoke(tool_call)` runs it and returns a `ToolMessage` with the right `tool_call_id`.

Lesson 2 does one round by hand. Often the model asks for one tool at a time, so one round isn't enough, and that's why you need a loop.

## 3. The agent: `create_agent` ([03_agent.py](03_agent.py))

```python
agent = create_agent(model="groq:openai/gpt-oss-120b", tools=[...], system_prompt="...")
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
result["messages"][-1].content      # final answer
```

```
        ┌─────────┐   tool_calls?   ┌─────────┐
  ───▶  │  model  │ ──── yes ─────▶ │  tools  │
        └─────────┘ ◀────────────── └─────────┘
             │ no
             ▼
            END
```
- It's built on **LangGraph**, so the agent is a small graph with two nodes.
- **Input and output are a *state* dict** with a `messages` list.
- `agent.stream(..., stream_mode="updates")` shows each node as it runs.
- **Memory:** `create_agent(..., checkpointer=InMemorySaver())`, then pass `{"configurable": {"thread_id": "abc"}}`. Same thread_id = same conversation.

## 4. Middleware ([04_middleware.py](04_middleware.py))

Middleware lets you change the hidden loop **without rewriting it**:

```
before_agent
  └─ loop:  before_model → [wrap_model_call → MODEL] → after_model
            → [wrap_tool_call → TOOL] → back to before_model
after_agent
```

| Hook | Use it for |
|---|---|
| `@before_model` / `@after_model` | Logging, trimming messages, checking output |
| `@wrap_model_call` | Retries, switching model, changing the request |
| `@wrap_tool_call` | Timing, error handling, approval, caching |
| `@dynamic_prompt` | System prompt built per user (from `context`) |

**Built-ins** used in the lesson:
- `ModelCallLimitMiddleware(run_limit=6)`: stops runaway loops
- `ToolCallLimitMiddleware(tool_name="calculator", run_limit=3)`: caps one tool
- `PIIMiddleware("email", strategy="redact")`: the model only ever sees `[REDACTED_EMAIL]`

Others worth knowing: `SummarizationMiddleware` (long chats), `HumanInTheLoopMiddleware` (approve before a tool runs), `ModelRetryMiddleware`, `ToolRetryMiddleware`, `ModelFallbackMiddleware`, `LLMToolSelectorMiddleware`, `TodoListMiddleware`.

**Context vs. state:**
- **State** (`messages`, …) changes during the run and is saved by the checkpointer.
- **Context** (`context_schema`, passed as `invoke(..., context=...)`) is read-only info about the run, such as the user's name, role or ID.

Middleware runs in **list order**; the first item is the outermost layer.

## 5. Structured output ([05_structured_output.py](05_structured_output.py))

Get **typed objects** instead of free text:

```python
class ContactInfo(BaseModel):
    name: str
    email: str | None = Field(None, description="Email if present")

model.with_structured_output(ContactInfo).invoke("...")       # one call, no tools
create_agent(..., response_format=ToolStrategy(TripComparison))  # tools first, then typed result
result["structured_response"]                                    # ← a validated Pydantic object
```

| Strategy | How it works | When |
|---|---|---|
| `ProviderStrategy` | Provider's native JSON-schema mode | Most reliable, if the model supports it |
| `ToolStrategy` | The schema becomes a "final answer" tool | Works with any tool-calling model |
| Pass the class directly | LangChain picks one | Default |

`Field(description=...)` matters for the same reason tool docstrings do: **the model reads it.**

---

## Exercises

1. **Switch providers:** get a free Gemini key, `pip install langchain-google-genai`, and change `MODEL_ID` in [config.py](config.py) to `"google_genai:gemini-2.5-flash"`. Nothing else should need to change.
2. **New tool:** add `convert_currency(amount: float, from_cur: str, to_cur: str)` with mock rates, then ask the agent a question that needs it plus the calculator.
3. **Human approval:** add `HumanInTheLoopMiddleware(interrupt_on={"calculator": True})` with a checkpointer and see the run pause (look up `Command(resume=...)`).
4. **Custom middleware:** write an `@after_model` that counts `usage_metadata["total_tokens"]` across the run and prints the total.
5. **Structured agent:** build a "recipe agent" whose `response_format` is a `Recipe(title, ingredients: list[str], steps: list[str], minutes: int)`.
6. **Capstone:** combine everything into a chat REPL like `01-agent-from-scratch/05_chat_agent.py`, with memory, middleware and a `/json` command that returns structured output.

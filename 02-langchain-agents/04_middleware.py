"""
LESSON 4: Middleware. Hook into the agent loop without rewriting it.

create_agent() hides the loop, so how do you add logging, limits, retries,
safety filters...? Middleware = functions that run at specific points:

   before_agent ─▶ ┌──────────────────────────────────────────────┐
                   │  before_model ─▶ [wrap_model_call: MODEL]    │
                   │                    ─▶ after_model            │
                   │       │ tool_calls?                          │
                   │       ▼                                      │
                   │  [wrap_tool_call: TOOL] ── loop back ──▶     │
                   └──────────────────────────────────────────────┘
                                         ─▶ after_agent

  before_* / after_*  : run before/after a step; can read or change the state
  wrap_*              : wrap the call itself (like a Python decorator): can
                        retry it, time it, change the request, or replace the result

Part A: built-in middleware (just configure it)
Part B: custom middleware (write your own with decorators)

Run:  python 04_middleware.py
"""

import time
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    after_model,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.messages import ToolMessage

from config import MODEL_ID, print_messages, title
from tools import ALL_TOOLS

# ==============================================================================
# Part B first: write custom middleware (Part A uses them together below)
# ==============================================================================


# 1) before_model: runs before EVERY model call. Here: simple logging.
@before_model
def log_model_call(state, runtime):
    print(f"  \033[90m[before_model] sending {len(state['messages'])} messages to the LLM\033[0m")
    return None  # None = don't change the state


# 2) after_model: runs after every model call. Here: report what the model decided.
@after_model
def log_decision(state, runtime):
    last = state["messages"][-1]
    decision = f"wants tools {[tc['name'] for tc in last.tool_calls]}" if last.tool_calls else "final answer"
    print(f"  \033[90m[after_model] model → {decision}\033[0m")
    return None


# 3) wrap_tool_call: wraps every tool execution. Here: timing + error handling.
@wrap_tool_call
def time_and_guard_tools(request, handler):
    name = request.tool_call["name"]
    start = time.perf_counter()
    try:
        result = handler(request)  # <- actually runs the tool
    except Exception as e:
        print(f"  \033[31m[wrap_tool_call] {name} FAILED: {e!r} → sending error to model\033[0m")
        # Turn a crash into a message the model can read and recover from
        # (the same idea as run_tool() in folder 01).
        return ToolMessage(content=f"Tool error: {e}. Fix your input and try again.",
                           tool_call_id=request.tool_call["id"], name=name)
    print(f"  \033[90m[wrap_tool_call] {name} took {(time.perf_counter() - start) * 1000:.1f} ms\033[0m")
    return result


# 4) dynamic_prompt: build the system prompt per request, e.g. from user context.
@dataclass
class UserContext:
    name: str
    expertise: str  # "beginner" or "expert"


@dynamic_prompt
def personalized_prompt(request):
    ctx = request.runtime.context
    style = "Explain simply, like to a beginner." if ctx.expertise == "beginner" else "Be terse and technical."
    return f"You are a helpful assistant talking to {ctx.name}. {style} Use tools for time, weather and math."


# ==============================================================================
# Build the agent: middleware runs in list order (first = outermost)
# ==============================================================================
agent = create_agent(
    model=MODEL_ID,
    tools=ALL_TOOLS,
    context_schema=UserContext,  # typed, read-only info passed in at invoke time
    middleware=[
        # --- Part A: built-ins ---
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),   # replaces our MAX_STEPS
        ToolCallLimitMiddleware(tool_name="calculator", run_limit=3),  # cap a specific tool
        PIIMiddleware("email", strategy="redact", apply_to_input=True),  # hide emails from the LLM
        # --- Part B: custom ---
        log_model_call,
        log_decision,
        time_and_guard_tools,
        personalized_prompt,
    ],
)

title("Run 1: beginner user, logs show each hook firing")
r = agent.invoke(
    {"messages": [{"role": "user", "content": "What's 2**10 and what's the weather in London?"}]},
    context=UserContext(name="Iris", expertise="beginner"),
)
print("\nAgent:", r["messages"][-1].content)

title("Run 2: PII middleware redacts the email BEFORE the model sees it")
r = agent.invoke(
    {"messages": [{"role": "user", "content": "Note my contact: iris.test@example.com. What text do you see where my email should be?"}]},
    context=UserContext(name="Iris", expertise="expert"),
)
print_messages(r["messages"])

title("Run 3: a tool error is caught by wrap_tool_call and handed back to the model")
r = agent.invoke(
    {"messages": [{"role": "user", "content": "What time is it in the timezone 'Mars/Olympus_Mons'? Call the time tool with exactly that string first; if it fails, use UTC instead."}]},
    context=UserContext(name="Iris", expertise="expert"),
)
print_messages(r["messages"])

# Other built-ins worth knowing (from langchain.agents.middleware):
#   SummarizationMiddleware  - summarizes old messages when the history gets too long
#   HumanInTheLoopMiddleware - pauses for human approval before chosen tools run
#   ModelRetryMiddleware / ToolRetryMiddleware - retry with backoff on errors
#   ModelFallbackMiddleware  - switch to a backup model if the main one fails
#   LLMToolSelectorMiddleware - lets a small LLM pick relevant tools when you have many
#   TodoListMiddleware       - gives the agent a planning/todo tool

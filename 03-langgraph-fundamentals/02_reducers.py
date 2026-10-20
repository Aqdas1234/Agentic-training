"""
LESSON 2: Reducers. HOW a node's update is merged into the state.

When a node returns {"key": value}, what happens to the old value?

  No reducer (default)          → OVERWRITE:  old value is replaced
  Annotated[list, operator.add] → APPEND:     old + new
  Annotated[list, add_messages] → append messages smartly (used for chat)

Why this matters:
  1. Collecting things across steps (logs, messages, results)
  2. PARALLEL nodes: if two nodes run at the same time and both write the
     same key, LangGraph needs a rule to combine them, or it raises an error.

Run:  python 02_reducers.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph, add_messages

from config import show_graph, title


# =============================================================================
# Part A: overwrite vs. append
# =============================================================================
class LogState(TypedDict):
    status: str                                # no reducer → overwritten
    log: Annotated[list[str], operator.add]    # reducer → appended


def step_one(state):
    return {"status": "step one done", "log": ["step one ran"]}


def step_two(state):
    return {"status": "step two done", "log": ["step two ran"]}


g = StateGraph(LogState)
g.add_sequence([step_one, step_two])  # same as add_node + add_edge for each
g.add_edge(START, "step_one")
g.add_edge("step_two", END)
result = g.compile().invoke({"status": "start", "log": ["input received"]})

title("A) overwrite vs. append")
print("status:", result["status"], "   ← only the LAST value survives")
print("log:   ", result["log"], "   ← every update was ADDED")


# =============================================================================
# Part B: parallel nodes (fan-out / fan-in)
# =============================================================================
#            ┌─▶ check_weather ─┐
#   START ───┤                   ├──▶ summarize ─▶ END
#            └─▶ check_news ────┘
# Both checks run in the SAME step (in parallel). Both write `findings`.

class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add]  # try removing the Annotated part!
    summary: str


def check_weather(state):
    return {"findings": [f"weather info about {state['topic']}"]}


def check_news(state):
    return {"findings": [f"news about {state['topic']}"]}


def summarize(state):
    return {"summary": f"{len(state['findings'])} findings: {state['findings']}"}


def build_research(state_cls):
    b = StateGraph(state_cls)
    b.add_node("check_weather", check_weather)
    b.add_node("check_news", check_news)
    b.add_node("summarize", summarize)
    b.add_edge(START, "check_weather")        # two edges from START
    b.add_edge(START, "check_news")           #   = run both in parallel
    b.add_edge(["check_weather", "check_news"], "summarize")  # wait for BOTH
    b.add_edge("summarize", END)
    return b.compile()


research = build_research(ResearchState)
show_graph(research)
title("B) parallel nodes WITH a reducer")
print(research.invoke({"topic": "Tokyo", "findings": []})["summary"])


# Same graph, but `findings` has NO reducer
class BrokenState(TypedDict):
    topic: str
    findings: list[str]
    summary: str


title("B) parallel nodes WITHOUT a reducer")
try:
    build_research(BrokenState).invoke({"topic": "Tokyo", "findings": []})
except InvalidUpdateError as e:
    print("  InvalidUpdateError:", str(e).splitlines()[0])
    print("  → Two nodes wrote `findings` in the same step and LangGraph can't choose.")


# =============================================================================
# Part C: add_messages, the reducer for chat history
# =============================================================================
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    # Shortcut: `from langgraph.graph import MessagesState` is exactly this.


def bot(state):
    last = state["messages"][-1].content
    return {"messages": [{"role": "assistant", "content": f"You said: {last}"}]}


chat = StateGraph(ChatState)
chat.add_node("bot", bot)
chat.add_edge(START, "bot")
chat.add_edge("bot", END)
out = chat.compile().invoke({"messages": [{"role": "user", "content": "hello"}]})

title("C) add_messages: appends AND converts dicts to message objects")
for m in out["messages"]:
    print(f"  {type(m).__name__:<13} id={m.id[:8]}…  {m.content!r}")
print("  (If you return a message with an EXISTING id, it replaces that message instead.)")

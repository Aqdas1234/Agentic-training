"""
LESSON 3: create_agent(). The whole agent loop in one function call.

Remember run_agent() in folder 01, lesson 4 (a while-loop of: call model ->
run tools -> append results -> repeat)? create_agent() IS that loop, built on
LangGraph (a library for building agents as graphs):

        ┌─────────┐   tool_calls?   ┌─────────┐
  ───▶  │  model  │ ──── yes ─────▶ │  tools  │
        └─────────┘ ◀────────────── └─────────┘
             │ no
             ▼
            END

You'll see:
  A) invoke(): run to completion, get the full message history back
  B) stream(): watch each step of the loop as it happens
  C) memory:   a checkpointer + thread_id remembers earlier turns

Run:  python 03_agent.py
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import MODEL_ID, print_messages, title
from tools import ALL_TOOLS

agent = create_agent(
    model=MODEL_ID,            # a "provider:model" string OR a model object
    tools=ALL_TOOLS,
    system_prompt="You are a helpful assistant. Use tools for time, weather and math; never guess.",
)

# ---- A) invoke: input and output are a STATE dict with a "messages" list --------
title("A) agent.invoke()")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo and what's 15% of 240?"}]}
)
print_messages(result["messages"])
print("\nFinal answer is just the last message:", result["messages"][-1].content[:80], "...")

# ---- B) stream: see each node (model / tools) as the loop runs ------------------
title("B) agent.stream() shows the loop step by step")
for step in agent.stream(
    {"messages": [{"role": "user", "content": "What time is it in London?"}]},
    stream_mode="updates",  # yields {node_name: state_update} after each step
):
    for node, update in step.items():
        print(f"\n\033[1m▶ node: {node}\033[0m")
        print_messages(update["messages"])

# ---- C) memory: checkpointer saves state per conversation (thread_id) -----------
title("C) Memory with a checkpointer")
agent_with_memory = create_agent(
    model=MODEL_ID,
    tools=ALL_TOOLS,
    checkpointer=InMemorySaver(),  # in-memory; use SqliteSaver/PostgresSaver to persist
)
config = {"configurable": {"thread_id": "iris-chat-1"}}  # like a chat session ID

agent_with_memory.invoke({"messages": [{"role": "user", "content": "Hi, I live in Mumbai."}]}, config)
r = agent_with_memory.invoke({"messages": [{"role": "user", "content": "What's the weather where I live?"}]}, config)
print("Agent:", r["messages"][-1].content)
print(f"(The thread has {len(r['messages'])} messages saved. We only sent 1 new message per call.)")

# In folder 01 YOU kept the messages list. Here the checkpointer does it,
# keyed by thread_id. A different thread_id = a fresh conversation.

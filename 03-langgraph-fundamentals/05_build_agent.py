"""
LESSON 5: Build a tool-calling agent from scratch in LangGraph.

This is what create_agent() (folder 02) builds for you. Now you'll build it
yourself using everything from lessons 1–4:

  STATE      MessagesState (messages + add_messages reducer) + a call counter
  NODES      "agent" (calls the LLM) and "tools" (runs the requested tools)
  EDGES      START → agent;  tools → agent   (← the LOOP)
  ROUTING    after agent: tool_calls? → "tools", otherwise → END
  COMPILE    with a checkpointer for memory

        START → agent ──tool_calls?──▶ tools
                  ▲ │                  │
                  │ └──no──▶ END       │
                  └────────────────────┘

Run:  python 05_build_agent.py
"""

from langchain.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from config import get_model, print_messages, show_graph, title
from tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
MAX_LLM_CALLS = 6

llm_with_tools = get_model().bind_tools(ALL_TOOLS)


# ---- STATE: extend MessagesState with our own field ---------------------------
class AgentState(MessagesState):
    llm_calls: int  # no reducer → overwritten each time (we pass the new total)


# ---- NODE 1: the agent (LLM) ---------------------------------------------------
def agent(state: AgentState) -> dict:
    system = SystemMessage("You are a helpful assistant. Use tools for time, weather and math; never guess.")
    response = llm_with_tools.invoke([system] + state["messages"])
    return {
        "messages": [response],                       # add_messages appends it
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# ---- NODE 2: the tools ---------------------------------------------------------
def tools(state: AgentState) -> dict:
    last = state["messages"][-1]                       # the AIMessage with tool_calls
    results = []
    for tc in last.tool_calls:
        try:
            output = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])
        except Exception as e:
            output = f"Error: {e}"                     # let the model recover
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": results}
    # Prebuilt shortcut: `from langgraph.prebuilt import ToolNode; ToolNode(ALL_TOOLS)`


# ---- ROUTER: continue the loop or stop? ---------------------------------------
def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if not last.tool_calls:
        return END                                     # final answer
    if state["llm_calls"] >= MAX_LLM_CALLS:
        return END                                     # safety limit
    return "tools"
    # Prebuilt shortcut: `from langgraph.prebuilt import tools_condition`


# ---- BUILD + COMPILE -----------------------------------------------------------
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")                     # ← THE LOOP

graph = builder.compile(checkpointer=InMemorySaver())
show_graph(graph)

# ---- RUN: stream to watch the loop ---------------------------------------------
config = {"configurable": {"thread_id": "lesson-5"}}

# NOTE: the checkpointer saves ALL state, including llm_calls. Without resetting it
# to 0 each turn, the counter would keep growing across turns and eventually hit
# MAX_LLM_CALLS. Each input can update any state key, so we reset it here.
title("Turn 1: streaming each node")
for step in graph.stream(
    {"messages": [{"role": "user", "content": "I'm in Paris. What's the weather, local time, and 18% of 85?"}],
     "llm_calls": 0},
    config,
    stream_mode="updates",
):
    for node, update in step.items():
        print(f"\n\033[1m▶ {node}\033[0m  (llm_calls={update.get('llm_calls', '-')})")
        print_messages(update["messages"])

title("Turn 2: memory via checkpointer (same thread_id)")
r = graph.invoke({"messages": [{"role": "user", "content": "Which city did I say I'm in?"}], "llm_calls": 0}, config)
print("  Agent:", r["messages"][-1].content)

title("Inspect saved state: graph.get_state(config)")
snapshot = graph.get_state(config)
print(f"  messages saved: {len(snapshot.values['messages'])}")
print(f"  llm_calls:      {snapshot.values['llm_calls']}   (this turn only, thanks to the reset)")
print(f"  next node:      {snapshot.next or '(finished)'}")

# Everything create_agent() does is here, with ~40 lines you fully control.
# Want an extra step (e.g. a "safety check" node before tools, or a "summarize"
# node at the end)? Just add_node + add_edge. That's why LangGraph exists.

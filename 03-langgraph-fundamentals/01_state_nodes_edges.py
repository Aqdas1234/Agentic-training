"""
LESSON 1: State, nodes, edges, compile. The 4 building blocks.

No LLM here on purpose: LangGraph is just a way to run Python functions in
a defined order while passing data between them. LLMs come later.

  STATE  = a shared dict that flows through the graph (defined with TypedDict)
  NODE   = a plain Python function:  state in  →  partial update out
  EDGE   = "after node A, run node B"
  COMPILE = check the graph is valid and turn it into something you can run

We build a tiny text pipeline:

    START → clean → count_words → make_report → END

Run:  python 01_state_nodes_edges.py
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import show_graph, title


# ---- 1. STATE: what data exists in the graph ---------------------------------
# Every node can READ all of it, and each node RETURNS only the keys it changes.
class PipelineState(TypedDict):
    text: str          # input
    cleaned: str       # set by `clean`
    word_count: int    # set by `count_words`
    report: str        # set by `make_report`


# ---- 2. NODES: plain functions (state) -> dict of updates --------------------
def clean(state: PipelineState) -> dict:
    print("  running node: clean")
    return {"cleaned": " ".join(state["text"].split()).strip().lower()}


def count_words(state: PipelineState) -> dict:
    print("  running node: count_words")
    return {"word_count": len(state["cleaned"].split())}


def make_report(state: PipelineState) -> dict:
    print("  running node: make_report")
    return {"report": f"'{state['cleaned']}' has {state['word_count']} words"}
    # Note: we never return the full state, only the keys this node owns.
    # LangGraph MERGES the update into the shared state.


# ---- 3. BUILD the graph: add nodes, then connect them with edges -------------
builder = StateGraph(PipelineState)        # "a graph whose state looks like this"

builder.add_node("clean", clean)           # name → function
builder.add_node("count_words", count_words)
builder.add_node("make_report", make_report)

builder.add_edge(START, "clean")           # START = where input enters
builder.add_edge("clean", "count_words")
builder.add_edge("count_words", "make_report")
builder.add_edge("make_report", END)       # END = where the run finishes
# Shortcut for straight lines:  builder.add_sequence([clean, count_words, make_report])

# ---- 4. COMPILE: validate + produce a runnable graph -------------------------
graph = builder.compile()
show_graph(graph)

# ---- RUN it ------------------------------------------------------------------
title("invoke(): pass the initial state, get the FINAL state back")
final = graph.invoke({"text": "   Hello    LangGraph   World  "})
print(final)

title("stream(mode='updates'): see what EACH node changed")
for update in graph.stream({"text": "Graphs are fun"}, stream_mode="updates"):
    print("  ", update)

title("stream(mode='values'): see the FULL state after each step")
for state in graph.stream({"text": "Graphs are fun"}, stream_mode="values"):
    print("  ", state)

# ---- What compile() protects you from -----------------------------------------
title("compile() catches mistakes BEFORE anything runs")
bad = StateGraph(PipelineState)
bad.add_node("clean", clean)
bad.add_edge(START, "clean")
bad.add_edge("clean", "count_wrods")      # typo: this node doesn't exist
try:
    bad.compile()
except ValueError as e:
    print("  ValueError:", e)

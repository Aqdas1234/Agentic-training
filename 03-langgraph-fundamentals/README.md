# 03 · LangGraph Fundamentals

In folder 02, `create_agent()` gave you an agent, but the loop was hidden. **LangGraph is the engine under it.** Here you learn to build any flow yourself: pipelines, routers, loops, and finally a full agent.

Topics: **state · nodes · edges · reducers · conditional routing · loops · compilation**

## Run
```bash
source ../.venv/bin/activate      # from this folder
pip install -r ../requirements.txt   # adds grandalf (draws graphs in the terminal)
python 01_state_nodes_edges.py    # no LLM
python 02_reducers.py             # no LLM
python 03_conditional_routing.py  # Part B uses the LLM
python 04_loops.py                # Part C uses the LLM
python 05_build_agent.py          # full agent
```

---

## Why LangGraph?
A plain LLM call is one step. Real AI apps are **flows**: classify → route → call tools → check → retry → answer. LangGraph lets you draw that flow as a **graph** and runs it for you, with state, memory, streaming, loops and pausing built in.

```
 Folder 01: you wrote the loop      →  full control, lots of code
 Folder 02: create_agent()          →  little code, fixed shape
 Folder 03: LangGraph StateGraph    →  little code, ANY shape you want
```

---

## The 4 building blocks ([01_state_nodes_edges.py](01_state_nodes_edges.py))

| Block | What it is | Code |
|---|---|---|
| **State** | A shared dict that flows through the graph | `class MyState(TypedDict): text: str` |
| **Node** | A Python function: reads state, returns **only the keys it changes** | `def clean(state): return {"cleaned": ...}` |
| **Edge** | "After A, run B" | `builder.add_edge("A", "B")` |
| **Compile** | Validates the graph and makes it runnable | `graph = builder.compile()` |

```python
builder = StateGraph(MyState)
builder.add_node("clean", clean)
builder.add_edge(START, "clean")
builder.add_edge("clean", END)
graph = builder.compile()
graph.invoke({"text": "hi"})     # → final state dict
```

- `START` and `END` are special built-in nodes for where input enters and where the run stops.
- **`compile()` checks the graph** for problems such as edges to missing nodes or nodes that can't be reached. It's also where you attach a `checkpointer` (memory) and `interrupt_before` (pauses).
- Running a graph:
  - `invoke()` returns the final state.
  - `stream(stream_mode="updates")` yields what each node changed.
  - `stream(stream_mode="values")` yields the full state after each step.

## Reducers: how updates merge ([02_reducers.py](02_reducers.py))

| State field | When a node returns a new value |
|---|---|
| `status: str` | **Overwrites** (the default) |
| `log: Annotated[list, operator.add]` | **Appends** (old + new) |
| `messages: Annotated[list, add_messages]` | Appends messages, converts dicts, **replaces by id** |

`MessagesState` is a ready-made state with `messages` + `add_messages`.

**Parallel nodes:** two edges out of the same node run both targets **in the same step**. If both write the same key, you **need** a reducer; otherwise you get `InvalidUpdateError`. `add_edge(["a", "b"], "c")` waits for both before running `c`.

## Conditional routing ([03_conditional_routing.py](03_conditional_routing.py))

```python
def route(state) -> str:                 # a router reads state and returns a node name
    return state["category"]

builder.add_conditional_edges("classify", route, ["billing", "tech", "general"])
```
- The router **must not change state**; it only decides where to go next.
- An **LLM can be the classifier**. Use `with_structured_output` with a `Literal[...]` type so its answer is always a valid node name.
- **Alternative: `Command`.** A node returns `Command(update={...}, goto="billing")`, so you don't need a separate router function. Add the return type `Command[Literal["billing", ...]]` so LangGraph knows the possible targets.

## Loops ([04_loops.py](04_loops.py))

A loop is a conditional edge that points **back** to an earlier node.

```
START → guess → check ──wrong──▶ guess      (loop)
                  └────right───▶ END        (exit)
```
**Every loop needs an exit.** Ways to guarantee one:
1. A condition in the router (`if score >= 8: return END`)
2. A counter in the state (`if revisions > 2: return END`)
3. `graph.invoke(input, {"recursion_limit": 10})` as a hard cap. If it's exceeded you get `GraphRecursionError`.

A **step** ("super-step") is one round of node runs. Parallel nodes share one step.

The **reflection pattern** (write → critique → revise) is a common LLM loop for improving output quality.

## Build an agent by hand ([05_build_agent.py](05_build_agent.py))

```
START → agent ──tool_calls?──▶ tools
          ▲ │                    │
          │ └──no──▶ END         │
          └──────────────────────┘
```
```python
class AgentState(MessagesState):
    llm_calls: int

builder.add_node("agent", call_llm)          # returns {"messages": [ai_msg]}
builder.add_node("tools", run_tools)         # returns {"messages": [tool_msgs]}
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")           # the loop
graph = builder.compile(checkpointer=InMemorySaver())
```
That's what `create_agent()` builds. LangGraph also has prebuilt pieces for it:
- `ToolNode(tools)` replaces your `tools` function.
- `tools_condition` replaces `should_continue`.

**Gotcha from this lesson:** a checkpointer saves **all** state, not just messages. Counters keep growing across turns unless you reset them in each input (`{"messages": [...], "llm_calls": 0}`).

**Inspecting state:** `graph.get_state(config)` shows the saved values and which node runs `next`. That's how you debug, and how human-in-the-loop pausing works.

The ASCII drawing doesn't show edges that loop back, such as `tools → agent`. For a full diagram, paste `graph.get_graph().draw_mermaid()` into https://mermaid.live.

---

## Summary

| Concept | One-liner |
|---|---|
| State | `TypedDict` describing the shared data |
| Node | `fn(state) -> partial update dict` |
| Edge | `add_edge(a, b)`: fixed next step |
| Reducer | `Annotated[type, fn]`: how updates merge |
| Conditional edge | `add_conditional_edges(src, router, targets)` |
| Command | Node returns `Command(update=..., goto=...)` |
| Loop | A conditional edge back to an earlier node, plus an exit |
| Compile | `builder.compile(checkpointer=...)`: validate and get a runnable graph |
| Run | `invoke` / `stream` / `get_state` |

## Exercises
1. **Reducer:** in `02_reducers.py`, add a third parallel node `check_events` and see that the reducer collects all 3 findings.
2. **Router:** add a `"refund"` category to lesson 3 with its own team node. Update the `Literal`, the node and the routes.
3. **Loop:** change `MAX_REVISIONS` and the score threshold in lesson 4. When does it exit via score, and when via the limit?
4. **Agent:** replace the hand-written `tools` node and `should_continue` in lesson 5 with `ToolNode` and `tools_condition`.
5. **New node:** add a `guardrail` node before `tools` in lesson 5 that blocks `calculator` expressions longer than 50 characters.
6. **Human in the loop:** compile lesson 5 with `interrupt_before=["tools"]`. Run it, check `graph.get_state(config).next`, then continue with `graph.invoke(None, config)`.

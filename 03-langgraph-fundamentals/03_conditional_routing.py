"""
LESSON 3: Conditional routing. Choosing the next node at runtime.

A normal edge always goes to the same place. A CONDITIONAL edge calls a
"router" function that looks at the state and RETURNS the name of the next node:

    builder.add_conditional_edges("classify", route_fn, ["billing", "tech", "general"])
                                   ^ from      ^ decides  ^ possible destinations

  Part A: rule-based router (keywords), no LLM
  Part B: LLM router: the model classifies, structured output makes it reliable
  Part C: Command, an alternative where the node itself says where to go next

Run:  python 03_conditional_routing.py
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from config import get_model, show_graph, title


class TicketState(TypedDict):
    ticket: str
    category: str
    reply: str


# ---- Handler nodes, shared by Parts A and B -----------------------------------
def billing_team(state):
    return {"reply": "💳 Billing team: we'll check your invoice within 24h."}


def tech_team(state):
    return {"reply": "🛠️ Tech team: please try restarting, then send us logs."}


def general_team(state):
    return {"reply": "💬 Support: thanks for reaching out, we'll reply soon."}


def build(classify_fn):
    b = StateGraph(TicketState)
    b.add_node("classify", classify_fn)
    b.add_node("billing", billing_team)
    b.add_node("tech", tech_team)
    b.add_node("general", general_team)

    b.add_edge(START, "classify")
    # The router function decides where to go after `classify`
    b.add_conditional_edges("classify", route_by_category, ["billing", "tech", "general"])
    for n in ("billing", "tech", "general"):
        b.add_edge(n, END)
    return b.compile()


# The ROUTER: reads state, returns a node name. It must NOT change the state.
def route_by_category(state) -> str:
    return state["category"]


# =============================================================================
# Part A: rule-based classification
# =============================================================================
def classify_with_rules(state):
    t = state["ticket"].lower()
    if any(w in t for w in ("refund", "invoice", "charged", "payment")):
        return {"category": "billing"}
    if any(w in t for w in ("error", "crash", "bug", "login")):
        return {"category": "tech"}
    return {"category": "general"}


tickets = [
    "I was charged twice this month!",
    "The app crashes when I open settings",
    "Do you have an office in Berlin?",
    "My money disappeared after I upgraded my plan",   # no keyword matches…
]

graph = build(classify_with_rules)
show_graph(graph)
title("A) Rule-based router")
for t in tickets:
    r = graph.invoke({"ticket": t})
    print(f"  [{r['category']:<7}] {t}\n            → {r['reply']}")
print("  ⚠️  The last ticket is really a billing issue, but no keyword matched.")


# =============================================================================
# Part B: LLM-based classification (understands meaning, not just keywords)
# =============================================================================
class Classification(BaseModel):
    category: Literal["billing", "tech", "general"] = Field(
        description="billing = money/payments/plans, tech = bugs/errors/login, general = anything else"
    )


# with_structured_output + Literal guarantees the answer is one of our node names
classifier = get_model(temperature=0).with_structured_output(Classification)


def classify_with_llm(state):
    result = classifier.invoke(f"Classify this support ticket: {state['ticket']}")
    return {"category": result.category}


title("B) LLM router (same graph, smarter classify node)")
llm_graph = build(classify_with_llm)
for t in tickets:
    r = llm_graph.invoke({"ticket": t})
    print(f"  [{r['category']:<7}] {t}")


# =============================================================================
# Part C: Command, which updates state AND picks the next node in one return
# =============================================================================
# Instead of a separate router function + add_conditional_edges, a node can
# return Command(update=..., goto=...). The Literal type hint tells LangGraph
# (and the graph drawing) which nodes it may jump to.
def classify_and_route(state) -> Command[Literal["billing", "tech", "general"]]:
    category = classify_with_rules(state)["category"]
    return Command(update={"category": category}, goto=category)


b = StateGraph(TicketState)
b.add_node("classify", classify_and_route)
b.add_node("billing", billing_team)
b.add_node("tech", tech_team)
b.add_node("general", general_team)
b.add_edge(START, "classify")          # no add_conditional_edges needed!
for n in ("billing", "tech", "general"):
    b.add_edge(n, END)
cmd_graph = b.compile()

title("C) Command-based routing")
print(" ", cmd_graph.invoke({"ticket": "Login error 500"})["reply"])

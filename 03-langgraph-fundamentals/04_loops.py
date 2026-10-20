"""
LESSON 4: Loops. Edges that point BACKWARDS.

A loop is a conditional edge whose router can send you back to an earlier
node. Every loop needs an EXIT CONDITION; otherwise it runs forever.

  Part A: a number-guessing loop (binary search), no LLM
  Part B: what happens with no exit → recursion_limit saves you
  Part C: an LLM "write → critique → revise" loop (a reflection agent)

Run:  python 04_loops.py
"""

from typing import TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import get_model, show_graph, title

# =============================================================================
# Part A: guessing game
# =============================================================================
#   START → guess → check ──"too high/low"──▶ guess   (loop back)
#                     └────"correct"───────▶ END


class GuessState(TypedDict):
    secret: int
    low: int
    high: int
    guess: int
    attempts: int
    feedback: str


def guess(state):
    g = (state["low"] + state["high"]) // 2
    return {"guess": g, "attempts": state["attempts"] + 1}


def check(state):
    g, s = state["guess"], state["secret"]
    print(f"  attempt {state['attempts']}: guessed {g:>3} (range {state['low']}–{state['high']})")
    if g < s:
        return {"feedback": "too low", "low": g + 1}
    if g > s:
        return {"feedback": "too high", "high": g - 1}
    return {"feedback": "correct"}


def keep_going(state) -> str:
    return END if state["feedback"] == "correct" else "guess"  # ← the exit condition


b = StateGraph(GuessState)
b.add_node("guess", guess)
b.add_node("check", check)
b.add_edge(START, "guess")
b.add_edge("guess", "check")
b.add_conditional_edges("check", keep_going, ["guess", END])
game = b.compile()

show_graph(game)
title("A) Loop until correct")
r = game.invoke({"secret": 73, "low": 1, "high": 100, "attempts": 0})
print(f"  ✅ Found {r['guess']} in {r['attempts']} attempts")


# =============================================================================
# Part B: a loop with no exit, stopped by recursion_limit
# =============================================================================
# LangGraph runs in "super-steps" (one round of nodes = one step).
# recursion_limit = the maximum number of steps per run. Pass it in the config.
class Counter(TypedDict):
    n: int


inf = StateGraph(Counter)
inf.add_node("tick", lambda s: {"n": s["n"] + 1})
inf.add_edge(START, "tick")
inf.add_edge("tick", "tick")  # always loops back, with no way out!
forever = inf.compile()

title("B) Infinite loop + recursion_limit=10")
try:
    forever.invoke({"n": 0}, {"recursion_limit": 10})
except GraphRecursionError:
    print("  GraphRecursionError: stopped after 10 steps (instead of running forever)")
print("  → In agents, ALWAYS set a limit: recursion_limit, a counter in state, or middleware.")


# =============================================================================
# Part C: reflection loop with an LLM
# =============================================================================
#   START → write → critique ──score < 8 and revisions left──▶ write (revise)
#                       └──────otherwise──────────────────────▶ END

MAX_REVISIONS = 2
llm = get_model(temperature=0.7)


class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="Quality from 1 to 10. Be strict: 8+ only if genuinely excellent.")
    feedback: str = Field(description="One concrete suggestion to improve it")


critic = get_model(temperature=0).with_structured_output(Critique)


class WriterState(TypedDict):
    topic: str
    draft: str
    score: int
    feedback: str
    revisions: int


def write(state):
    if not state.get("draft"):
        prompt = f"Write a 2-line poem about {state['topic']}. Output only the poem."
    else:
        prompt = (f"Improve this 2-line poem using the feedback. Output only the poem.\n"
                  f"Poem:\n{state['draft']}\nFeedback: {state['feedback']}")
    draft = llm.invoke(prompt).content.strip()
    print(f"\n  ✍️  draft v{state.get('revisions', 0) + 1}:\n     " + draft.replace("\n", "\n     "))
    return {"draft": draft, "revisions": state.get("revisions", 0) + 1}


def critique(state):
    c = critic.invoke(f"Critique this 2-line poem about {state['topic']}:\n{state['draft']}")
    print(f"  🧐 score {c.score}/10: {c.feedback}")
    return {"score": c.score, "feedback": c.feedback}


def should_revise(state) -> str:
    if state["score"] >= 8:
        print("  → good enough, stop")
        return END
    if state["revisions"] > MAX_REVISIONS:
        print("  → out of revisions, stop")
        return END
    print("  → revise again")
    return "write"


b = StateGraph(WriterState)
b.add_node("write", write)
b.add_node("critique", critique)
b.add_edge(START, "write")
b.add_edge("write", "critique")
b.add_conditional_edges("critique", should_revise, ["write", END])
writer = b.compile()

show_graph(writer)
title("C) Reflection loop: write → critique → revise")
final = writer.invoke({"topic": "a programmer learning about AI agents", "revisions": 0})
print(f"\n  Final (score {final['score']}, {final['revisions']} draft(s)):")
print("  " + final["draft"].replace("\n", "\n  "))

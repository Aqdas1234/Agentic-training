"""
LESSON 4: The agent loop. This is what makes something an "agent".

Lesson 3 did exactly ONE round of tool use. Real tasks may need several:
the model calls a tool, looks at the result, decides to call another, and so on.
So we wrap lesson 3 in a loop:

    while True:
        response = model(messages, tools)
        if no tool_calls:  -> done, return the text
        else:              -> run every tool, append results, loop again

Safety: MAX_STEPS stops runaway loops (a model can keep calling tools forever).

Run:  python 04_agent_loop.py
"""

import json

from config import MODEL, assistant_to_dict, client
from tools import TOOL_SCHEMAS, run_tool

MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a helpful assistant with tools. "
    "Use tools whenever they help; never guess times, weather or math. "
    "You may call several tools at once. When you have everything, answer clearly."
)


def run_agent(user_input: str, messages: list | None = None, verbose: bool = True) -> str:
    """Run the tool loop until the model gives a final answer."""
    if messages is None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_input})

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # let the model decide: "auto" | "none" | "required"
        )
        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        if verbose:
            print(f"\n\033[33m--- step {step}: finish_reason={finish} ---\033[0m")

        # Always record what the model said, including its tool requests.
        messages.append(assistant_to_dict(msg))

        # ----- No tool calls -> the model is done -------------------------------
        if not msg.tool_calls:
            return msg.content

        # ----- Tool calls -> run each one and feed the results back ------------
        # The model may request SEVERAL tools in one turn ("parallel tool calls").
        for call in msg.tool_calls:
            result = run_tool(call.function.name, call.function.arguments)
            if verbose:
                args = json.loads(call.function.arguments or "{}")
                print(f"  🔧 {call.function.name}({args})  ->  {result}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # loop: the model sees the results on the next iteration

    return "Stopped: reached MAX_STEPS without a final answer."


if __name__ == "__main__":
    question = (
        "What time is it in Tokyo right now, and what's the weather there? "
        "Also, what is 15% of 240?"
    )
    print(type(question))
    print(f"USER: {question}")
    answer = run_agent(question)
    print(f"\n\033[1;32mAGENT:\033[0m {answer}")

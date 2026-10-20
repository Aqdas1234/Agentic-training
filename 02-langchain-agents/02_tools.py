"""
LESSON 2: Tools. Define them with @tool and bind them to a model.

Three things to understand:
  1. @tool turns a Python function into a tool (name, description, schema).
     Look at tools.py, then run `python tools.py` to see the generated schemas.
  2. model.bind_tools([...]) returns a NEW model that knows about the tools.
     (Like passing tools=TOOL_SCHEMAS in folder 01.)
  3. AIMessage.tool_calls is a normalized list of dicts:
        [{"name": "calculator", "args": {"expression": "..."}, "id": "call_..."}]
     Note: `args` is already a parsed dict, so no more json.loads()!

This lesson still does the tool round trip BY HAND (like folder 01, lesson 3),
so you can see exactly what create_agent() automates in the next lesson.

Run:  python 02_tools.py
"""

from langchain.messages import HumanMessage

from config import get_model, title
from tools import ALL_TOOLS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

model = get_model()
model_with_tools = model.bind_tools(ALL_TOOLS)

messages = [HumanMessage("What is 1234 * 5678, and what's the weather in Paris?")]

# ---- Step 1: model decides which tools to call ---------------------------------
ai_msg = model_with_tools.invoke(messages)
title("Step 1: AIMessage.tool_calls (the model's requests)")
for tc in ai_msg.tool_calls:
    print(tc)
messages.append(ai_msg)

# ---- Step 2: run each tool -----------------------------------------------------
# Calling tool.invoke(tool_call) with the WHOLE tool_call dict returns a ready-made
# ToolMessage with the correct tool_call_id. That's the step we built by hand before.
title("Step 2: run tools → ToolMessages")
for tc in ai_msg.tool_calls:
    tool_msg = TOOLS_BY_NAME[tc["name"]].invoke(tc)
    print(f"{type(tool_msg).__name__}(tool_call_id={tool_msg.tool_call_id!r}, content={tool_msg.content!r})")
    messages.append(tool_msg)

# ---- Step 3: model writes the final answer ---------------------------------------
final = model_with_tools.invoke(messages)
if final.tool_calls:
    # The model may call tools one at a time over several turns instead of all
    # at once. Handling ANY number of rounds needs a LOOP, which is lesson 3.
    title("Step 3: the model wants MORE tools, so one round wasn't enough")
    for tc in final.tool_calls:
        print(tc)
    print("\n→ This is exactly why agents need a loop. See 03_agent.py.")
else:
    title("Step 3: final answer")
    print(final.content)

"""
LESSON 3: One tool call, done by hand, step by step.

The round trip:
  1. We send messages + the list of tools (JSON schemas).
  2. The model replies with NO text, but with `tool_calls`:
        [{id: "call_abc", function: {name: "calculator", arguments: '{"expression":"1234*5678"}'}}]
     and finish_reason = "tool_calls".
  3. WE run the function in Python.
  4. We append TWO things to messages:
        a) the assistant message that contains the tool_calls (exactly as returned)
        b) a {"role": "tool", "tool_call_id": "call_abc", "content": "<result>"} message
     The tool_call_id links each result to the request it answers.
  5. We call the model again, and it now writes a normal answer using the result.

Run:  python 03_single_tool_call.py
"""

import json

from config import MODEL, assistant_to_dict, client, show
from tools import TOOL_SCHEMAS, run_tool

# Use only the calculator for this lesson to keep things focused.
tools = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "calculator"]

messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools for math."},
    {"role": "user", "content": "What is 1234 * 5678?"},
]
show("STEP 1: the tool definition the model sees", tools)

# ---- STEP 2: first call, the model decides to use a tool --------------------
response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
assistant_msg = response.choices[0].message
show("STEP 2: model response (finish_reason)", response.choices[0].finish_reason)
show("STEP 2: assistant message (note: content is empty, tool_calls is set)", assistant_msg)

if not assistant_msg.tool_calls:
    print("\nThe model answered without a tool this time:", assistant_msg.content)
    raise SystemExit

# (a) Append the assistant's tool request to the history, exactly as returned.
messages.append(assistant_to_dict(assistant_msg))

# ---- STEP 3 & 4: run each requested tool and append its result ---------------
for call in assistant_msg.tool_calls:
    name = call.function.name
    args = call.function.arguments          # a JSON *string*
    result = run_tool(name, args)
    show(f"STEP 3: we ran {name}({json.loads(args)})", result)

    # (b) Append the result, linked by tool_call_id.
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

show("STEP 4: messages now", messages)

# ---- STEP 5: second call, the model uses the result to answer ----------------
final = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
show("STEP 5: final finish_reason", final.choices[0].finish_reason)
show("STEP 5: final answer", final.choices[0].message.content)

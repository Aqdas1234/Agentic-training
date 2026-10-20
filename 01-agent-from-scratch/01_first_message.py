"""
LESSON 1: Send one message and look at what comes back.

Key ideas:
  - You send a LIST of messages. Each message is {"role": ..., "content": ...}.
  - The model is stateless: it only knows what is in that list.
  - The answer is in response.choices[0].message.
  - finish_reason tells you WHY the model stopped:
        "stop"       -> it finished its answer normally
        "length"     -> it hit max_tokens
        "tool_calls" -> it wants you to run a tool (lesson 3)
  - usage tells you how many tokens were used (what paid APIs bill for).

Run:  python 01_first_message.py
"""

from config import MODEL, client, show

messages = [
    {"role": "user", "content": "In one sentence, what is an AI agent?"},
]
show("What we SEND (messages)", messages)

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0.3,   # lower = more predictable, higher = more creative
    max_tokens=200,    # hard cap on the length of the reply
)

show("RAW response object (everything the API returns)", response)

choice = response.choices[0]
show("The assistant message", choice.message)
show("finish_reason", choice.finish_reason)
show("Token usage", response.usage)
show("Just the text", choice.message.content)

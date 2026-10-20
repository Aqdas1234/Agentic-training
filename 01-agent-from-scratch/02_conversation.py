"""
LESSON 2: Roles and conversation "memory".

The three basic roles:
  system    -> instructions/personality for the model (sent first)
  user      -> what the human says
  assistant -> what the model said before

The model has NO memory between API calls. "Memory" is simply us
appending every message to a list and re-sending the WHOLE list each time.
That is also why long chats cost more tokens: the list keeps growing.

Run:  python 02_conversation.py
"""

from config import MODEL, client, show

messages = [
    {"role": "system", "content": "You are a friendly tutor. Keep answers under 2 sentences."},
]


def ask(user_text: str) -> str:
    messages.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(model=MODEL, messages=messages)
    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})
    print(f"\nYou: {user_text}\nAI : {reply}  [prompt tokens: {response.usage.prompt_tokens}]")
    return reply


ask("Hi! My name is Iris and I'm learning about AI agents.")
ask("What's a good first project for me?")
ask("What's my name?")  # works only because turn 1 is still in `messages`

show("The full messages list we re-sent on the last call", messages)
print("\nNotice that prompt tokens grow every turn, because the whole history is sent again each time.")

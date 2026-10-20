"""
LESSON 5: An interactive chat agent in your terminal.

It combines everything:
  - conversation memory (lesson 2): one `messages` list kept across turns
  - tools (lesson 3)
  - the agent loop (lesson 4): reuses run_agent()

Try:
  > what files are in this project?
  > read config.py and tell me which model I'm using
  > what's 2**32 divided by 7, and what time is it in London?
  > /verbose     (toggle showing the tool steps)
  > /history     (print the raw messages list)
  > exit

Run:  python 05_chat_agent.py
"""

import importlib

from config import MODEL, show

# Python files starting with a digit can't be imported normally, so use importlib.
run_agent = importlib.import_module("04_agent_loop").run_agent
SYSTEM_PROMPT = importlib.import_module("04_agent_loop").SYSTEM_PROMPT


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    verbose = True
    print(f"Chat agent using {MODEL}. Type 'exit' to quit, '/verbose' or '/history'.")

    while True:
        try:
            user_input = input("\n\033[1mYou >\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input == "/verbose":
            verbose = not verbose
            print(f"verbose = {verbose}")
            continue
        if user_input == "/history":
            show("messages", messages)
            continue

        try:
            answer = run_agent(user_input, messages, verbose=verbose)
        except Exception as e:  # e.g. rate limit on the free tier
            print(f"API error: {e}")
            continue
        print(f"\n\033[1;32mAgent >\033[0m {answer}")

    print("Bye!")


if __name__ == "__main__":
    main()

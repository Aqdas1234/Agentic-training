# Agentic Training

Learn how AI agents work, from raw API calls up to frameworks.

| Folder | What you learn |
|---|---|
| [01-agent-from-scratch](01-agent-from-scratch/README.md) | Messages, model responses, tool calling and the agent loop, built by hand with the `openai` SDK |
| [02-langchain-agents](02-langchain-agents/README.md) | The same agent in LangChain: model integrations, `@tool`, `create_agent`, middleware, structured output |
| [03-langgraph-fundamentals](03-langgraph-fundamentals/README.md) | The engine under LangChain: state, nodes, edges, reducers, conditional routing, loops, compilation, and an agent built by hand |

## Setup (once)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # paste your free Groq key: https://console.groq.com/keys
```
Then `cd` into a folder and run its lessons in order.

"""
tools.py: the same tools as folder 01, rewritten the LangChain way.

In folder 01 every tool needed: a function + a hand-written JSON schema
+ a registry entry + a run_tool() dispatcher.

With LangChain, the @tool decorator generates all of that from:
  - the function NAME      -> tool name
  - the DOCSTRING          -> description the model reads
  - the TYPE HINTS         -> JSON schema for the arguments
  - Annotated[...] / Field -> per-argument descriptions
"""

import ast
import json
import operator
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from langchain.tools import tool

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


@tool
def calculator(expression: Annotated[str, "Arithmetic expression, e.g. '1234*5678' or '0.15*240'"]) -> str:
    """Evaluate an arithmetic expression exactly. Use this for ANY math."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and + - * / ** % // are allowed")

    return str(_eval(ast.parse(expression, mode="eval").body))


@tool
def get_current_time(timezone: Annotated[str, "IANA timezone, e.g. 'Asia/Tokyo'"]) -> str:
    """Get the current date and time in a timezone."""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature in Celsius and conditions."""
    fake = {
        "tokyo": {"temp_c": 18, "condition": "cloudy"},
        "london": {"temp_c": 11, "condition": "rainy"},
        "new york": {"temp_c": 15, "condition": "sunny"},
        "mumbai": {"temp_c": 31, "condition": "humid"},
        "paris": {"temp_c": 13, "condition": "windy"},
    }
    data = fake.get(city.lower(), {"temp_c": 22, "condition": "clear"})
    return json.dumps({"city": city, **data, "note": "mock data"})


ALL_TOOLS = [calculator, get_current_time, get_weather]


if __name__ == "__main__":
    # See what @tool generated. No API key needed:  python tools.py
    for t in ALL_TOOLS:
        print(f"\nname:        {t.name}")
        print(f"description: {t.description}")
        print(f"args schema: {json.dumps(t.args)}")
    print("\nDirect call:", calculator.invoke({"expression": "1234*5678"}))

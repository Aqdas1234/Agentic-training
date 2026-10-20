"""
tools.py: the "hands" of our agent.

A tool has TWO parts:
  1. A normal Python function that does the work.
  2. A JSON Schema description that we send to the model, so it knows the
     tool exists, what it does, and which arguments it takes.

The model NEVER runs code itself. It only *asks* us to run a tool by
returning its name + JSON arguments. Our code runs it and sends the result back.
"""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).parent.resolve()


# =========================================================================
# 1) The Python functions
# =========================================================================

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic like '1234*5678' or '0.15*240' (no eval!)."""

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError("Only numbers and + - * / ** % // are allowed")

    return str(_eval(ast.parse(expression, mode="eval").body))


def get_current_time(timezone: str = "UTC") -> str:
    """Current date/time in an IANA timezone such as 'Asia/Tokyo'."""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_weather(city: str) -> str:
    """FAKE weather so the lesson needs no extra API key."""
    fake = {
        "tokyo": {"temp_c": 18, "condition": "cloudy"},
        "london": {"temp_c": 11, "condition": "rainy"},
        "new york": {"temp_c": 15, "condition": "sunny"},
        "mumbai": {"temp_c": 31, "condition": "humid"},
    }
    data = fake.get(city.lower(), {"temp_c": 22, "condition": "clear"})
    return json.dumps({"city": city, **data, "note": "mock data"})


def _safe_path(path: str) -> Path:
    """Keep file tools inside the project folder (basic safety guard)."""
    p = (PROJECT_DIR / path).resolve()
    if PROJECT_DIR not in (p, *p.parents):
        raise ValueError("Access outside the project folder is not allowed")
    return p


def list_files(directory: str = ".") -> str:
    """List files in a folder inside the project."""
    d = _safe_path(directory)
    names = sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir()
                   if not x.name.startswith("."))
    return "\n".join(names)


def read_file(path: str) -> str:
    """Read a text file inside the project (first 4000 chars)."""
    return _safe_path(path).read_text()[:4000]


# =========================================================================
# 2) The JSON-schema descriptions the MODEL sees
#    Good descriptions matter: the model decides WHEN to call a tool
#    based only on `description` text.
# =========================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression exactly. Use for ANY math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '1234*5678' or '0.15*240'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in a timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA name, e.g. 'Asia/Tokyo', 'Europe/London'"},
                },
                "required": ["timezone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory of the project.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string", "description": "Relative path, '.' for root"}},
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file in the project.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    },
]

# name -> function, so we can look up what the model asked for
TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "list_files": list_files,
    "read_file": read_file,
}


def run_tool(name: str, arguments_json: str) -> str:
    """
    Execute a tool the model asked for.

    The model sends arguments as a JSON *string*, so we parse it first.
    On ANY error we return the error text instead of crashing: the model
    reads it and can usually fix its call on the next step.
    """
    try:
        func = TOOL_REGISTRY[name]
        args = json.loads(arguments_json or "{}")
        return str(func(**args))
    except KeyError:
        return f"Error: unknown tool '{name}'"
    except Exception as e:
        return f"Error running {name}: {type(e).__name__}: {e}"


if __name__ == "__main__":
    # Quick self-test, no API needed:  python tools.py
    print(run_tool("calculator", '{"expression": "1234*5678"}'))
    print(run_tool("get_current_time", '{"timezone": "Asia/Tokyo"}'))
    print(run_tool("get_weather", '{"city": "Tokyo"}'))
    print(run_tool("list_files", '{"directory": "."}'))
    print(run_tool("read_file", '{"path": "../../etc/passwd"}'))  # should be blocked
    print(run_tool("nope", "{}"))

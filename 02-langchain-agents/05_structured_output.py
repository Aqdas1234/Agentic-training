"""
LESSON 5: Structured output. Get typed data back, not free text.

Free text is great for humans but bad for code. If your program needs
fields (to save to a DB, show in a UI, pass to another function), ask the
model to fill a SCHEMA. Define it with Pydantic:

    class Movie(BaseModel):
        title: str
        year: int

LangChain makes sure the result is a validated `Movie` object.

Two places to use it:
  A) model.with_structured_output(Schema): a single model call, no tools
  B) create_agent(..., response_format=Schema): the agent uses tools first,
     then returns the structured result in result["structured_response"]

Two strategies behind the scenes (for create_agent):
  ProviderStrategy: uses the provider's native "JSON schema mode" (most reliable)
  ToolStrategy:     turns the schema into a fake "tool" that the model calls
                    with its answer; works with ANY model that supports tools
  Passing the class directly lets LangChain pick automatically.

Run:  python 05_structured_output.py
"""

from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from config import MODEL_ID, get_model, title
from tools import ALL_TOOLS


# ---- Schemas: Field(description=...) tells the model what to put in each field ----
class ContactInfo(BaseModel):
    """Contact details extracted from text."""
    name: str
    email: str | None = Field(None, description="Email if present")
    phone: str | None = Field(None, description="Phone number if present")
    company: str | None = None


class CityReport(BaseModel):
    """Weather and time report for one city."""
    city: str
    local_time: str = Field(description="Current local time from the time tool")
    temp_c: float
    condition: str
    clothing_tip: str = Field(description="One short sentence on what to wear")


class TripComparison(BaseModel):
    """Compare cities for a trip."""
    reports: list[CityReport]
    best_city: str = Field(description="The city with the nicest weather right now")
    mood: Literal["great", "okay", "stay home"]


# ---- A) Structured output from a single model call (extraction) ------------------
title("A) model.with_structured_output(): extract data from text")
extractor = get_model(temperature=0).with_structured_output(ContactInfo)
contact = extractor.invoke(
    "Hey, it's Priya Sharma from Acme Corp. Reach me at priya@acme.io or +91 98765 43210."
)
print(type(contact).__name__, "→", contact)
print("Access fields like a normal object:", contact.name, "|", contact.company)

# ---- B) Agent that uses tools, then returns structured data -----------------------
title("B) create_agent(response_format=...): tools + typed result")
agent = create_agent(
    model=MODEL_ID,
    tools=ALL_TOOLS,
    system_prompt="Use the time and weather tools for every city before answering.",
    response_format=ToolStrategy(TripComparison),  # works with any tool-calling model
)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Compare Tokyo, London and Mumbai for a trip right now."}]}
)
trip: TripComparison = result["structured_response"]

for r in trip.reports:
    print(f"  {r.city:<8} {r.local_time:<26} {r.temp_c:>5}°C  {r.condition:<8} → {r.clothing_tip}")
print(f"  Best city: {trip.best_city}  |  mood: {trip.mood}")

title("It's a real Pydantic object, so convert it to JSON in one line")
print(trip.model_dump_json(indent=2)[:400], "...")

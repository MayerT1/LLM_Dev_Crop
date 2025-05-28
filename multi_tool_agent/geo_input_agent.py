from typing import Optional
from google.adk.agents import Agent

def parse_geo_input(geo: dict, time_period: Optional[dict] = None) -> dict:
    """Validates and returns geo and time range."""
    if not geo or "type" not in geo or "coordinates" not in geo:
        return {
            "status": "error",
            "error_message": "Invalid or missing geo object. Must include 'type' and 'coordinates'."
        }

    # Use default time range if not provided
    time_period = time_period or {"start": "2023-01-01", "end": "2023-12-31"}

    return {
        "status": "success",
        "geo": geo,
        "time_period": time_period
    }

geo_input_agent = Agent(
    name="geo_input_agent",
    model="gemini-2.0-flash",
    description="Validates user-supplied geography and time period.",
    instruction="You are responsible for checking that user inputs are valid geography formats.",
    tools=[parse_geo_input],
)

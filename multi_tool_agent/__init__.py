from .geo_input_agent import parse_geo_input
from .geo_data_agent import get_geo_data
from .geo_market_agent import estimate_crop_price
from google.adk.agents import Agent

agent = Agent(
    name="geo_market_multi_agent",
    model="gemini-2.0-flash",
    description="Handles geospatial input, Earth Engine data retrieval, and market price estimation.",
    instruction="You take user location and time input, fetch geospatial crop, soil, and weather data, and estimate the market price for the identified crop.",
    tools=[parse_geo_input, get_geo_data, estimate_crop_price],
)

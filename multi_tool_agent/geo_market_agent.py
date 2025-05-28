from google.adk.agents import Agent
import random

CROP_CODE_MAP = {
    1: "Corn", 5: "Soybeans", 24: "Durum Wheat", 26: "Winter Wheat",
    23: "Barley", 3: "Rice", 4: "Sorghum", 6: "Sunflower"
}

CROP_PRICES = {
    "Corn": 4.6,
    "Soybeans": 13.1,
    "Wheat": 6.9,
    "Barley": 5.2
}

def estimate_crop_price(crop_type: dict, time_period: dict) -> dict:
    """Estimates price for a given crop type over time."""
    try:
        crop_code = int(crop_type.get("cropland", 0))
        crop_name = CROP_CODE_MAP.get(crop_code, "Unknown")
        price = CROP_PRICES.get(crop_name, round(random.uniform(5, 12), 2))

        return {
            "status": "success",
            "crop": crop_name,
            "estimated_price_usd_per_bushel": price,
            "time_period": time_period
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

market_agent = Agent(
    name="market_agent",
    model="gemini-2.0-flash",
    description="Estimates commodity prices based on crop type and time period.",
    instruction="Use crop codes to estimate prices based on known or simulated market data.",
    tools=[estimate_crop_price],
)

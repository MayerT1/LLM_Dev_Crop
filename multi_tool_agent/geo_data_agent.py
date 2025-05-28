import ee
from google.adk.agents import Agent

# Set your actual Google Cloud Project ID here
EE_PROJECT_ID = "servir-sco-assets"

def initialize_ee():
    try:
        ee.Initialize(project=EE_PROJECT_ID)
    except Exception as e:
        raise RuntimeError(f"Earth Engine initialization failed: {e}")

def get_geo_data(geo: dict, time_period: dict) -> dict:
    """Retrieves crop type, soil moisture, and precipitation for a location and time range."""
    try:
        initialize_ee()

        geometry = ee.Geometry(geo)
        start = time_period["start"]
        end = time_period["end"]

        soil = ee.ImageCollection("NASA_USDA/HSL/SMAP10KM_soil_moisture") \
            .filterDate(start, end).mean().select("ssm")

        precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
            .filterDate(start, end).sum().select("precipitation")

        crops = ee.ImageCollection("USDA/NASS/CDL") \
            .filterDate(start, end).first().select("cropland")

        soil_moisture = soil.reduceRegion(ee.Reducer.mean(), geometry, 10000).getInfo()
        precipitation = precip.reduceRegion(ee.Reducer.sum(), geometry, 5000).getInfo()
        crop_type = crops.reduceRegion(ee.Reducer.mode(), geometry, 30).getInfo()

        return {
            "status": "success",
            "soil_moisture": soil_moisture,
            "precipitation": precipitation,
            "crop_type": crop_type,
        }

    except Exception as e:
        return {"status": "error", "error_message": str(e)}

geo_data_agent = Agent(
    name="geo_data_agent",
    model="gemini-2.0-flash",
    description="Fetches soil moisture, precipitation, and crop type for a location and time range using Earth Engine.",
    instruction="You are responsible for retrieving geospatial data for a given area and time.",
    tools=[get_geo_data],
)

# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:39:02 2025

@author: tjmayer
"""

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import yfinance as yf
import datetime


############
import geopandas as gpd
import pandas as pd
import numpy as np


# Authenticate and initialize Earth Engine
import ee
import geemap
import geemap.chart as chart
ee.Authenticate()
ee.Initialize(project='servir-sco-assets')
Map = geemap.Map()



#Coordinates for the bounds of a rectangle.
xMin = -89.63038298978414;
yMin = 35.92644038827682;
xMax = -89.43400237454976;
yMax = 36.070319808111755;

#Construct a rectangle from a list of GeoJSON 'point' formatted coordinates.
ROI= ee.Geometry.Rectangle(xMin, yMin,xMax, yMax)
Map.addLayer(ROI, {}, 'ROI');


NASS = ee.ImageCollection('USDA/NASS/CDL').filter(ee.Filter.date('2018-01-01', '2019-12-31')).first().clip(ROI)
cropLandcover = NASS.select('cropland');
Map.addLayer(cropLandcover, {}, 'Crop Landcover');


Map
print("ROI has been loaded")

################


# === Load open-source LLM ===
model_id = "microsoft/phi-2"  # Fast, small
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

print("mole loaded")

# === Helper functions ===

def fetch_soil_data(input_str: str) -> str:
    return "Soil data: pH 6.4, OM 2.9%, Type: silty clay loam"

def fetch_precip_data(input_str: str) -> str:
    today = datetime.date.today().isoformat()
    return f"Precipitation on {today}: 6.7 mm"

def fetch_yield_data(cultivar: str) -> str:
    yield_est = np.random.normal(3200, 200)
    return f"Estimated yield for {cultivar}: {yield_est:.2f} kg/ha"

def fetch_crop_price(input_str: str) -> str:
    try:
        price = yf.download('ZS=F', period="5d", interval="1d").iloc[-1]["Close"]
    except Exception:
        price = 530.0  # fallback
    return f"Soybean futures price: ${price:.2f}/ton"

def economic_eval(input_str: str) -> str:
    # Expected format: "cultivar,yield_kg,price_per_ton"
    try:
        cultivar, yield_kg, price = input_str.split(",")
        income = float(yield_kg) / 1000 * float(price)
        return f"Economic value for {cultivar.strip()}: ${income:.2f}/ha"
    except:
        return "Error in economic evaluation. Input should be 'cultivar,yield_kg,price'"

# === Define LangChain tools ===

tools = [
    Tool(name="SoilData", func=fetch_soil_data, description="Get soil data for the region."),
    Tool(name="PrecipData", func=fetch_precip_data, description="Fetch daily rainfall."),
    Tool(name="YieldEstimator", func=fetch_yield_data, description="Estimate yield for a soybean cultivar."),
    Tool(name="PriceFetcher", func=fetch_crop_price, description="Fetch soybean market price per ton."),
    Tool(name="EconEvaluator", func=economic_eval, description="Calculate economic return from yield and price. Input format: 'cultivar,yield_kg,price'")
]

# === Initialize agent with output parsing errors handled ===

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # ✅ important!
)

print("Run the workflow")
# === Run the workflow ===

def run_crop_eval(cultivar: str):
    prompt = (
        f"Evaluate soybean cultivar '{cultivar}' grown in western Tennessee. "
        "Get soil and precipitation data, estimate yield, fetch crop price, "
        "and compute economic value in dollars per hectare."
    )
    result = agent.run(prompt)
    print("\n✅ Final Result:\n", result)

# === Example usage ===
run_crop_eval("Asgrow AG38X8")

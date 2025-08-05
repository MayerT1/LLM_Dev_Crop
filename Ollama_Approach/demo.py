#!/usr/bin/env python3
"""
Demo script for the Economic Evaluator Agent
Demonstrates the agent's capabilities with example scenarios
"""

import sys
from pathlib import Path
import traceback

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from economic_evaluator import EconomicEvaluatorAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_complete_query():
    """Demo with a complete query that has all required information"""
    print("\n" + "=" * 60)
    print("DEMO 1: Complete Query (No Clarification Needed)")
    print("=" * 60)

    agent = EconomicEvaluatorAgent()

    query = """
    I want to plant maize on 2024-03-15 using a short season variety. 
    My fertilization plan is 100 kg/ha nitrogen at planting and 50 kg/ha at 30 days after planting.
    """

    print(f"User Query: {query.strip()}")
    print("\nProcessing...")

    try:
        result = agent.evaluate(query)
        print(f"\nAgent Response:\n{result}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")


def demo_incomplete_query():
    """Demo with incomplete query requiring clarification"""
    print("\n" + "=" * 60)
    print("DEMO 2: Incomplete Query (Clarification Loop)")
    print("=" * 60)

    agent = EconomicEvaluatorAgent()

    # Mock the clarification to avoid interactive input
    original_ask_clarification = agent._ask_clarification_node

    def mock_clarification(state):
        """Provide automatic clarification for demo purposes"""
        print("\n[DEMO] Agent would ask for clarification here...")
        print(f"[DEMO] Missing information: {', '.join(state.missing_info)}")

        # Simulate user providing missing information
        additional_info = ""
        if any("date" in info for info in state.missing_info):
            additional_info += " Planting date: 2024-04-01"
        if any("fertilization" in info for info in state.missing_info):
            additional_info += " Apply 75 kg/ha nitrogen at planting"
        if any("season" in info or "length" in info for info in state.missing_info):
            additional_info += " Use medium season variety"

        print(f"[DEMO] User provides: {additional_info}")

        combined_query = f"{state.user_query} {additional_info}"
        return {"user_query": combined_query}

    agent._ask_clarification_node = mock_clarification

    query = "I want to grow corn with some fertilizer"

    print(f"User Query: {query}")
    print("\nProcessing...")

    try:
        result = agent.evaluate(query)
        print(f"\nFinal Agent Response:\n{result}")
    except Exception as e:
        print(f"Error: {e}")


def demo_parsing_capabilities():
    """Demo the parsing capabilities of different components"""
    print("\n" + "=" * 60)
    print("DEMO 3: Parsing Capabilities")
    print("=" * 60)

    agent = EconomicEvaluatorAgent()
    parser = agent.parser

    # Test date parsing
    print("\n--- Date Parsing ---")
    date_examples = [
        "2024-03-15",
        "Plant on 3/15/2024",
        "Planting date is March 15, 2024"
    ]

    for example in date_examples:
        result = parser.parse_date(example)
        print(f"'{example}' -> {result}")

    # Test season parsing
    print("\n--- Season Parsing ---")
    season_examples = [
        "short season variety",
        "we want very long season",
        "medium length cultivar"
    ]

    for example in season_examples:
        result = parser.parse_planting_length(example)
        cultivar = agent.parser.parse_planting_length(example)
        if cultivar:
            from economic_evaluator import CULTIVAR_MAPPING
            cultivar_code = CULTIVAR_MAPPING.get(cultivar)
            print(f"'{example}' -> {result} -> {cultivar_code}")
        else:
            print(f"'{example}' -> {result}")


def demo_experiment_output():
    """Demo the structure of experiment output"""
    print("\n" + "=" * 60)
    print("DEMO 4: Experiment Output Structure")
    print("=" * 60)

    from economic_evaluator import run_experiment

    # Run a sample experiment
    sample_fert_plan = [[100, 0], [50, 30]]  # 100 kg/ha at planting, 50 kg/ha at 30 days

    print("Sample Parameters:")
    print(f"  Planting Date: 2024-03-15")
    print(f"  Fertilization Plan: {sample_fert_plan}")
    print(f"  Cultivar: KY0015 (short season)")
    print(f"  Location: St. Clair, Alabama")

    try:
        growth_phases, stress_factors, yield_results = run_experiment(
            planting_date="2024-03-15",
            fert_plan=sample_fert_plan,
            cultivar="KY0015",
            admin1_country="alabama",
            admin1_name="St. Clair"
        )

        print("\nExperiment Results:")
        print(f"\nGrowth Phases: {growth_phases}")
        print(f"\nStress Factors: {stress_factors}")
        print(f"\nYield Results: {yield_results}")

    except Exception as e:
        print(f"Error running experiment: {e}")


def main():
    """Run all demos"""
    print("Economic Evaluator Agent - Demo")
    print("===============================")
    print("This demo showcases the capabilities of the Economic Evaluator Agent")
    print("for agricultural experiment evaluation.")

    try:
        # Run demos
        demo_parsing_capabilities()
        demo_experiment_output()
        demo_complete_query()
        demo_incomplete_query()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

        print("\nTo run the agent interactively, use:")
        print("python economic_evaluator.py")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")


if __name__ == "__main__":
    main()
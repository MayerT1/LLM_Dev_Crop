"""
Example usage of the Economic Evaluator Agent
Demonstrates different scenarios and workflows
"""

from datetime import datetime, timedelta
from agent import EconomicEvaluatorAgent, CropData, FertilizationPlan


def example_scenario_1():
    """Scenario 1: User has yield data and wants to compute economic value"""
    print("=== SCENARIO 1: User with Yield Data ===")

    agent = EconomicEvaluatorAgent()

    # Example queries for this scenario
    queries = [
        "I have harvested 12.5 tons of wheat and want to know the market value",
        "I got 8.2 tons of corn, what's it worth?",
        "My soybeans yielded 15 tons, can you calculate the economic value?"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = agent.process_query(query)
        print(f"Agent: {response}")
        print("-" * 50)


def example_scenario_2():
    """Scenario 2: User has planted crop with fertilization plan"""
    print("\n=== SCENARIO 2: User with Fertilization Plan ===")

    agent = EconomicEvaluatorAgent()

    # Example queries for this scenario
    queries = [
        "I planted wheat 45 days ago and have this fertilization plan: day 30 with 60 kg/ha nitrogen, day 60 with 40 kg/ha nitrogen. What's the expected market value?",
        "I have corn planted 60 days ago with fertilization at day 20 (50 kg/ha) and day 45 (45 kg/ha). Calculate expected economics.",
        "Rice planted 30 days ago, fertilized at day 15 with 55 kg/ha nitrogen. What yield and value can I expect?"
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = agent.process_query(query)
        print(f"Agent: {response}")
        print("-" * 50)


def example_scenario_3():
    """Scenario 3: User has planted crop but no fertilization plan"""
    print("\n=== SCENARIO 3: User without Fertilization Plan ===")

    agent = EconomicEvaluatorAgent()

    # Example queries for this scenario
    queries = [
        "I planted soybeans 20 days ago but haven't fertilized yet. Can you help me evaluate different fertilization options?",
        "I have barley planted 35 days ago with no fertilization plan. What are my options?",
        "Wheat planted 15 days ago, no fertilization done yet. Show me different scenarios."
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = agent.process_query(query)
        print(f"Agent: {response}")
        print("-" * 50)


def interactive_mode():
    """Interactive mode for testing"""
    print("\n=== INTERACTIVE MODE ===")
    print("Enter your queries or type 'quit' to exit")

    agent = EconomicEvaluatorAgent()

    while True:
        user_input = input("\nYour query: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input:
            try:
                response = agent.process_query(user_input)
                print(f"\nAgent: {response}")
            except Exception as e:
                print(f"Error: {e}")


def test_tool_functions():
    """Test the individual tool functions"""
    print("\n=== TESTING TOOL FUNCTIONS ===")

    from agent import EconomicEvaluatorTools

    tools = EconomicEvaluatorTools()

    # Test market price lookup
    print("Testing market price lookup:")
    crops = ["wheat", "corn", "soybeans", "rice"]
    for crop in crops:
        price = tools.find_market_price(crop)
        print(f"  {crop}: ${price:.2f}/ton")

    # Test yield estimation
    print("\nTesting yield estimation:")
    planting_date = datetime.now() - timedelta(days=45)
    fertilization_plan = [
        FertilizationPlan(20, 60),
        FertilizationPlan(45, 40)
    ]

    for crop in crops:
        yield_estimate = tools.estimate_yield(crop, planting_date, fertilization_plan)
        print(f"  {crop}: {yield_estimate:.2f} tons")

    # Test market value calculation
    print("\nTesting market value calculation:")
    for crop in crops:
        price = tools.find_market_price(crop)
        yield_est = tools.estimate_yield(crop, planting_date, fertilization_plan)
        market_value = tools.calculate_market_value(yield_est, price)
        print(f"  {crop}: {yield_est:.2f} tons × ${price:.2f}/ton = ${market_value:.2f}")


if __name__ == "__main__":
    print("Economic Evaluator Agent - Example Usage")
    print("=" * 50)

    # Run example scenarios
    example_scenario_1()
    example_scenario_2()
    example_scenario_3()

    # Test tool functions
    test_tool_functions()

    # Interactive mode
    interactive_mode()
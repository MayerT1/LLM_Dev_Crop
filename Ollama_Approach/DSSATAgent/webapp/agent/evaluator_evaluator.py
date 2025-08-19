import sys
import json
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from economic_evaluator import EconomicEvaluatorAgent, ExperimentParser, CULTIVAR_MAPPING
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestEconomicEvaluator:
    """Test suite for the Economic Evaluator Agent"""

    def __init__(self):
        self.agent = EconomicEvaluatorAgent()
        self.parser = self.agent.parser

    def test_date_parsing(self):
        """Test date parsing functionality"""
        print("\n=== Testing Date Parsing ===")

        test_cases = [
            ("Plant on 2024-03-15", "2024-03-15"),
            ("Planting date: 3/15/2024", "2024-03-15"),
            ("We planted on March 15, 2024", None),  # This might not parse without more complex NLP
            ("2024-03-15 is the date", "2024-03-15"),
        ]

        for test_input, expected in test_cases:
            result = self.parser.parse_date(test_input)
            status = "✓" if result == expected else "✗"
            print(f"{status} Input: '{test_input}' -> Output: {result} (Expected: {expected})")

    def test_fertilization_parsing(self):
        """Test fertilization plan parsing"""
        print("\n=== Testing Fertilization Parsing ===")

        test_cases = [
            "100 kg/ha nitrogen at planting, 50 kg/ha at 30 days",
            "Apply 75 kg N/ha at day 0 and 25 kg N/ha at day 45",
            "No fertilizer",
            "200 kg/ha at planting only"
        ]

        for test_input in test_cases:
            result = self.parser.parse_fertilization_plan(test_input)
            print(f"Input: '{test_input}' -> Output: {result}")

    def test_planting_length_parsing(self):
        """Test planting length parsing"""
        print("\n=== Testing Planting Length Parsing ===")

        test_cases = [
            ("short season variety", "short season"),
            ("We want a very long season", "very long season"),
            ("medium length cultivar", "medium"),
            ("quick growing short variety", "short"),
        ]

        for test_input, expected in test_cases:
            result = self.parser.parse_planting_length(test_input)
            status = "✓" if result == expected else "✗"
            print(f"{status} Input: '{test_input}' -> Output: {result} (Expected: {expected})")

    def test_complete_scenarios(self):
        """Test complete evaluation scenarios"""
        print("\n=== Testing Complete Scenarios ===")

        scenarios = [
            {
                "name": "Complete Query",
                "query": "Plant on 2024-03-15 with 100 kg/ha nitrogen at planting and 50 kg/ha at 30 days, using short season variety"
            },
            {
                "name": "Missing Date",
                "query": "Apply 100 kg/ha nitrogen at planting, short season variety"
            }
        ]

        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            print(f"Query: {scenario['query']}")

            try:
                # Test just the parsing without full workflow to avoid interactive prompts
                state_data = {}
                state_data.update(self.agent._parse_input_node(type('State', (), {'user_query': scenario['query']})()))

                missing = []
                if not state_data.get('planting_date'):
                    missing.append("planting date")
                if not state_data.get('fertilization_plan'):
                    missing.append("fertilization plan")
                if not state_data.get('cultivar'):
                    missing.append("cultivar")

                print(f"Parsed data: {state_data}")
                print(f"Missing: {missing}")

            except Exception as e:
                print(f"Error: {e}")

    def test_cultivar_mapping(self):
        """Test cultivar mapping"""
        print("\n=== Testing Cultivar Mapping ===")

        for length, cultivar in CULTIVAR_MAPPING.items():
            print(f"{length} -> {cultivar}")

    def run_all_tests(self):
        """Run all test suites"""
        print("Economic Evaluator Agent - Test Suite")
        print("=" * 50)

        try:
            self.test_date_parsing()
            self.test_planting_length_parsing()
            self.test_fertilization_parsing()
            self.test_cultivar_mapping()
            self.test_complete_scenarios()

            print("\n" + "=" * 50)
            print("Test suite completed!")

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            print(f"Test suite failed: {e}")


def run_interactive_test():
    """Run interactive test with predefined scenarios"""
    print("\nInteractive Test Mode")
    print("=" * 30)

    agent = EconomicEvaluatorAgent()

    # Override the clarification node to avoid interactive input during testing
    original_ask_clarification = agent._ask_clarification_node

    def mock_ask_clarification(state):
        """Mock clarification that provides missing info automatically"""
        print(f"Mock clarification - Missing: {state.missing_info}")

        # Provide mock responses for missing information
        additional_info = ""
        if "planting date" in str(state.missing_info):
            additional_info += " Planting date: 2024-03-15"
        if "fertilization" in str(state.missing_info):
            additional_info += " Fertilization: 100 kg/ha at planting"
        if "planting length" in str(state.missing_info) or "season" in str(state.missing_info):
            additional_info += " Short season variety"

        combined_query = f"{state.user_query} {additional_info}"
        return {"user_query": combined_query}

    agent._ask_clarification_node = mock_ask_clarification

    test_queries = [
        "Plant on 2024-03-15 with 100 kg/ha nitrogen at planting, short season",
        "We need to test a medium season variety planted in spring",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        print(f"Query: {query}")

        try:
            result = agent.evaluate(query)
            print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run unit tests
    test_suite = TestEconomicEvaluator()
    test_suite.run_all_tests()

    # Ask if user wants to run interactive tests
    print(f"\nWould you like to run interactive tests? (y/n): ", end="")
    response = input().lower()

    if response in ['y', 'yes']:
        run_interactive_test()
    else:
        print("Skipping interactive tests.")
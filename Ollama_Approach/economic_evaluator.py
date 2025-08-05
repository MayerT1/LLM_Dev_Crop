import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from langchain_ollama import OllamaLLM
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np

from dssat import run_experiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("config.json not found, using default configuration")
        return {"ollama_url": "http://localhost:11434"}


config = load_config()


@dataclass
class ExperimentState:
    """State for the experiment evaluation workflow"""
    user_query: str = ""
    planting_date: Optional[str] = None
    fertilization_plan: Optional[List[List[float]]] = None
    planting_length: Optional[str] = None
    cultivar: Optional[str] = None
    missing_info: List[str] = None
    experiment_results: Optional[Dict] = None
    natural_language_output: str = ""
    messages: List = None
    needs_clarification: bool = False

    def __post_init__(self):
        if self.missing_info is None:
            self.missing_info = []
        if self.messages is None:
            self.messages = []


# Cultivar mappings
CULTIVAR_MAPPING = {
    "very short": "KY0017",
    "very short season": "KY0017",
    "short": "KY0015",
    "short season": "KY0015",
    "medium": "KY0002",
    "medium season": "KY0002",
    "long": "KY0013",
    "long season": "KY0013",
    "very long": "KY0012",
    "very long season": "KY0012"
}


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


@tool
def experiment_tool(planting_date: str, fert_plan: List[List[float]], cultivar: str) -> Dict:
    """Tool to run the agricultural experiment"""
    try:
        growth_phases, stress_factors, yield_results = run_experiment(
            planting_date=planting_date,
            fert_plan=fert_plan,
            cultivar=cultivar,
            admin1_country="Alabama",
            admin1_name="St. Clair"
        )

        # Convert any numpy types to native Python types for serialization
        result = {
            "growth_phases": convert_numpy_types(growth_phases),
            "stress_factors": convert_numpy_types(stress_factors),
            "yield_results": convert_numpy_types(yield_results)
        }

        return result
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return {"error": str(e)}


class ExperimentParser:
    """Parses user input to extract experiment parameters"""

    def __init__(self, llm):
        self.llm = llm

    def parse_date(self, text: str) -> Optional[str]:
        """Extract and validate planting date"""
        logger.info(f"Parsing date from: {text}")

        # Try different date patterns
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{1,2}/\d{1,2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}-\d{1,2}-\d{4})',  # MM-DD-YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                try:
                    if '/' in date_str:
                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                    elif '-' in date_str and len(date_str.split('-')[0]) <= 2:
                        date_obj = datetime.strptime(date_str, '%m-%d-%Y')
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

                    result = date_obj.strftime('%Y-%m-%d')
                    logger.info(f"Parsed date: {result}")
                    return result
                except ValueError:
                    continue

        logger.warning("No valid date found")
        return None

    def parse_fertilization_plan(self, text: str) -> Optional[List[List[float]]]:
        """Parse fertilization plan from text"""
        logger.info(f"Parsing fertilization plan from: {text}")

        # Use LLM to help parse complex fertilization plans
        prompt = f"""
        Extract the fertilization plan from this text: "{text}"

        Look for information about:
        - Nitrogen rates (in kg/ha or similar units)
        - Days after planting when fertilizer is applied

        Return the plan as a list of [nitrogen_rate, days_after_planting] pairs.
        If you find mentions like "100 kg/ha at planting", that means days_after_planting = 0.
        If you find "50 kg/ha at 30 days", that means days_after_planting = 30.

        Format your response EXACTLY as: [[rate1, days1], [rate2, days2], ...]
        Use only numbers, no units or extra text.
        If no fertilization plan is found, respond with "NONE_FOUND"

        Example: [[100, 0], [50, 30]]
        """

        try:
            response = self.llm.invoke(prompt)
            logger.info(f"LLM fertilization parsing response: {response}")

            if "NONE_FOUND" in response:
                return None

            # Clean the response and try multiple extraction approaches
            cleaned_response = response.strip()

            # Try to extract the array from the response using multiple patterns
            import ast

            # Pattern 1: Look for array-like patterns (more permissive with whitespace and newlines)
            array_patterns = [
                r'\[\s*\[[\d\s\.,\[\]]+\]\s*\]',  # Main nested array pattern with flexible whitespace
                r'\[[\[\],\s\d\.\n]+\]',  # Original pattern with newlines
                r'\[\s*(?:\[[\d\s\.,]+\]\s*,?\s*)+\]',  # Flexible nested arrays
            ]

            result = None
            for pattern in array_patterns:
                array_match = re.search(pattern, cleaned_response, re.DOTALL)
                if array_match:
                    try:
                        # Clean up the matched string
                        matched_str = array_match.group()
                        # Remove extra whitespace and newlines
                        matched_str = re.sub(r'\s+', ' ', matched_str)

                        logger.info(f"Attempting to parse: {matched_str}")
                        result = ast.literal_eval(matched_str)

                        # Validate format
                        if isinstance(result, list) and all(
                                isinstance(item, list) and len(item) == 2 for item in result):
                            # Convert to floats
                            result = [[float(item[0]), float(item[1])] for item in result]
                            logger.info(f"Successfully parsed fertilization plan: {result}")
                            return result
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse with pattern {pattern}: {parse_error}")
                        continue

            # Pattern 2: Try to find individual [rate, days] pairs and combine them
            pair_pattern = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
            pairs = re.findall(pair_pattern, cleaned_response)

            if pairs:
                try:
                    result = [[float(rate), float(days)] for rate, days in pairs]
                    logger.info(f"Parsed fertilization plan from pairs: {result}")
                    return result
                except Exception as pair_error:
                    logger.warning(f"Failed to parse pairs: {pair_error}")

            # Pattern 3: Try to extract numbers and infer structure
            numbers = re.findall(r'\d+(?:\.\d+)?', cleaned_response)
            if len(numbers) >= 2 and len(numbers) % 2 == 0:
                try:
                    result = []
                    for i in range(0, len(numbers), 2):
                        result.append([float(numbers[i]), float(numbers[i + 1])])

                    logger.info(f"Inferred fertilization plan from numbers: {result}")
                    return result
                except Exception as number_error:
                    logger.warning(f"Failed to infer from numbers: {number_error}")

            logger.warning("Could not parse fertilization plan from LLM response")
            return None

        except Exception as e:
            logger.error(f"Error parsing fertilization plan: {e}")
            return None

    def parse_planting_length(self, text: str) -> Optional[str]:
        """Extract planting length/season"""
        logger.info(f"Parsing planting length from: {text}")

        text_lower = text.lower()

        for length_key in CULTIVAR_MAPPING.keys():
            if length_key in text_lower:
                logger.info(f"Found planting length: {length_key}")
                return length_key

        logger.warning("No planting length found")
        return None


class EconomicEvaluatorAgent:
    """Main agent for economic evaluation"""

    def __init__(self):
        self.llm = OllamaLLM(
            base_url=config["ollama_url"],
            model="llama3.1:latest",
            verbose=True
        )
        self.parser = ExperimentParser(self.llm)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ExperimentState)

        # Add nodes
        workflow.add_node("parse_input", self._parse_input_node)
        workflow.add_node("check_completeness", self._check_completeness_node)
        workflow.add_node("ask_clarification", self._ask_clarification_node)
        workflow.add_node("run_experiment", self._run_experiment_node)
        workflow.add_node("generate_output", self._generate_output_node)

        # Define the flow
        workflow.set_entry_point("parse_input")

        workflow.add_edge("parse_input", "check_completeness")

        workflow.add_conditional_edges(
            "check_completeness",
            self._should_ask_clarification,
            {
                "clarify": "ask_clarification",
                "proceed": "run_experiment"
            }
        )

        workflow.add_edge("ask_clarification", "parse_input")
        workflow.add_edge("run_experiment", "generate_output")
        workflow.add_edge("generate_output", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _parse_input_node(self, state: ExperimentState) -> Dict:
        """Parse the user input to extract parameters"""
        logger.info("=== PARSE INPUT NODE ===")
        logger.info(f"Processing query: {state.user_query}")

        text = state.user_query

        # Parse each component
        planting_date = self.parser.parse_date(text)
        fertilization_plan = self.parser.parse_fertilization_plan(text)
        planting_length = self.parser.parse_planting_length(text)

        # Convert planting length to cultivar
        cultivar = None
        if planting_length:
            cultivar = CULTIVAR_MAPPING.get(planting_length)

        logger.info(
            f"Parsed - Date: {planting_date}, Fert: {fertilization_plan}, Length: {planting_length}, Cultivar: {cultivar}")

        return {
            "planting_date": planting_date,
            "fertilization_plan": fertilization_plan,
            "planting_length": planting_length,
            "cultivar": cultivar
        }

    def _check_completeness_node(self, state: ExperimentState) -> Dict:
        """Check if all required information is present"""
        logger.info("=== CHECK COMPLETENESS NODE ===")

        missing = []

        if not state.planting_date:
            missing.append("planting date (format: YYYY-MM-DD)")

        if not state.fertilization_plan:
            missing.append("fertilization plan (nitrogen rates and application timing)")

        if not state.planting_length or not state.cultivar:
            missing.append("planting length/season (very short, short, medium, long, or very long)")

        needs_clarification = len(missing) > 0

        logger.info(f"Missing information: {missing}")
        logger.info(f"Needs clarification: {needs_clarification}")

        return {
            "missing_info": missing,
            "needs_clarification": needs_clarification
        }

    def _should_ask_clarification(self, state: ExperimentState) -> str:
        """Determine if clarification is needed"""
        return "clarify" if state.needs_clarification else "proceed"

    def _ask_clarification_node(self, state: ExperimentState) -> Dict:
        """Ask for clarification on missing information"""
        logger.info("=== ASK CLARIFICATION NODE ===")

        if not state.missing_info:
            return {"user_query": state.user_query}

        missing_str = ", ".join(state.missing_info)

        clarification_prompt = f"""
I need more information to run the experiment. The following information is missing or unclear:

{missing_str}

Please provide the missing information. Here are some examples:

For planting date: "2024-03-15" or "March 15, 2024"
For fertilization plan: "100 kg/ha nitrogen at planting, 50 kg/ha at 30 days after planting"
For planting length: "short season" or "medium season" or "long season"
"""

        logger.info(f"Asking for clarification: {clarification_prompt}")
        print(f"\nAgent: {clarification_prompt}")

        # Get user input
        user_response = input("\nYou: ")
        logger.info(f"User clarification response: {user_response}")

        # Combine original query with clarification
        combined_query = f"{state.user_query} {user_response}"

        return {"user_query": combined_query}

    def _run_experiment_node(self, state: ExperimentState) -> Dict:
        """Run the agricultural experiment"""
        logger.info("=== RUN EXPERIMENT NODE ===")

        try:
            results = experiment_tool.invoke({
                "planting_date": state.planting_date,
                "fert_plan": state.fertilization_plan,
                "cultivar": state.cultivar
            })

            logger.info(f"Experiment results: {results}")

            return {"experiment_results": results}

        except Exception as e:
            logger.error(f"Error running experiment: {e}")
            return {"experiment_results": {"error": str(e)}}

    def _generate_output_node(self, state: ExperimentState) -> Dict:
        """Generate natural language output from results"""
        logger.info("=== GENERATE OUTPUT NODE ===")

        if not state.experiment_results or "error" in state.experiment_results:
            error_msg = state.experiment_results.get("error",
                                                     "Unknown error") if state.experiment_results else "No results"
            return {"natural_language_output": f"Error running experiment: {error_msg}"}

        results = state.experiment_results

        # Create a comprehensive prompt for natural language generation
        prompt = f"""
        Generate a natural language summary of the agricultural experiment results. Make it informative and easy to understand.

        Experiment Parameters:
        - Planting Date: {state.planting_date}
        - Cultivar: {state.cultivar} ({state.planting_length} season)
        - Fertilization Plan: {state.fertilization_plan}
        - Location: St. Clair, Alabama

        Results:
        Growth Phases: {results.get('growth_phases', {})}
        Stress Factors: {results.get('stress_factors', {})}
        Yield Results: {results.get('yield_results', {})}

        Please provide:
        1. A summary of the experiment setup
        2. Key findings from the growth phases and stress factors
        3. Yield predictions with confidence intervals
        4. Economic implications or recommendations

        Make the response conversational and accessible to farmers and agricultural professionals.
        
        Unless there is explicit mention of phosphorous assume that the fertilization plan only uses Nitrogen.
        """

        logger.info("Sending prompt to LLM for natural language generation")
        logger.info(f"Prompt: {prompt}")

        try:
            response = self.llm.invoke(prompt)
            logger.info(f"LLM Response: {response}")
            return {"natural_language_output": response}
        except Exception as e:
            logger.error(f"Error generating natural language output: {e}")
            return {"natural_language_output": f"Error generating summary: {e}"}

    def evaluate(self, user_query: str) -> str:
        """Main entry point for evaluation"""
        logger.info(f"Starting evaluation for query: {user_query}")

        initial_state = ExperimentState(user_query=user_query)

        # Run the workflow
        config_dict = {"configurable": {"thread_id": "1"}}

        final_state_dict = {}
        for state in self.workflow.stream(initial_state, config_dict):
            logger.info(f"Workflow step: {list(state.keys())}")
            # Update our tracking of the final state
            for node_name, node_output in state.items():
                if isinstance(node_output, dict):
                    final_state_dict.update(node_output)

        # Extract the natural language output
        result = final_state_dict.get('natural_language_output', 'Error: No output generated')

        if not result or result == 'Error: No output generated':
            # Fallback: try to construct a basic response from available data
            if final_state_dict.get('experiment_results'):
                result = f"Experiment completed successfully. Results: {str(final_state_dict['experiment_results'])[:200]}..."
            else:
                result = "Error: No output generated"

        logger.info(f"Final result: {result}")
        return result


def main():
    """Interactive main function for testing"""
    print("Economic Evaluator Agent")
    print("========================")
    print("Enter your experiment query (or 'quit' to exit)")

    agent = EconomicEvaluatorAgent()

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        try:
            result = agent.evaluate(user_input)
            print(f"\nAgent: {result}")

        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
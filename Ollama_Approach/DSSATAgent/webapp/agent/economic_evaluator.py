import json
import re
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from difflib import get_close_matches

from langchain_ollama import OllamaLLM
from langchain.tools import tool
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import numpy as np

from .dssat import run_experiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
def load_config():
    return {
        'ollama_url': os.environ.get('OLLAMA_URL'),
        'model_name': os.environ.get('MODEL_NAME')
    }


config = load_config()


@dataclass
class ExperimentState:
    """State for the experiment evaluation workflow"""
    user_query: str = ""
    planting_date: Optional[str] = None
    fertilization_plan: Optional[List[List[float]]] = None
    planting_length: Optional[str] = None
    cultivar: Optional[str] = None
    county: Optional[str] = None
    location_exact_match: bool = False
    location_valid: bool = True
    location_error_message: Optional[str] = None
    missing_info: List[str] = None
    experiment_results: Optional[Dict] = None
    natural_language_output: str = ""
    messages: List = None
    needs_clarification: bool = False
    should_end_conversation: bool = False

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
def experiment_tool(planting_date: str, fert_plan: List[List[float]], cultivar: str, county: str) -> Dict:
    """Tool to run the agricultural experiment"""
    try:
        print(planting_date, fert_plan, cultivar, county)
        yield_results, water_stress, nitro_stress = run_experiment(
            planting_date=planting_date,
            fert_plan=fert_plan,
            cultivar=cultivar,
            admin1_country="alabama",
            admin1_name=county
        )

        # Convert any numpy types to native Python types for serialization
        result = {
            "yield_results": convert_numpy_types(yield_results),
            "water_stress": convert_numpy_types(water_stress),
            "nitro_stress": convert_numpy_types(nitro_stress)
        }

        return result
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return {"error": str(e)}


class ExperimentParser:
    """Parses user input to extract experiment parameters"""

    def __init__(self, llm):
        self.llm = llm
        self.county_data = self._load_county_data("county_data.json")
        self.alabama_counties = [county["admin1"] for county in self.county_data]

    def _load_county_data(self, path: str) -> List[Dict]:
        """Load county data from JSON file"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} counties from {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading county data: {e}")
            return []

    def parse_location(self, text: str) -> Dict:
        """Extract and validate location (state/country and county)"""
        logger.info(f"Parsing location from: {text}")

        text_lower = text.lower()

        # Check for country mentions
        country_keywords = ['usa', 'united states', 'america', 'us']
        state_keywords = ['alabama', 'al']
        non_alabama_states = [
            'california', 'texas', 'florida', 'new york', 'georgia', 'illinois',
            'pennsylvania', 'ohio', 'michigan', 'north carolina', 'tennessee',
            'virginia', 'washington', 'arizona', 'massachusetts', 'indiana',
            'colorado', 'maryland', 'missouri', 'wisconsin', 'minnesota',
            'louisiana', 'south carolina', 'kentucky', 'oregon', 'oklahoma',
            'connecticut', 'utah', 'iowa', 'nevada', 'arkansas', 'mississippi',
            'kansas', 'new mexico', 'nebraska', 'west virginia', 'idaho',
            'hawaii', 'new hampshire', 'maine', 'montana', 'rhode island',
            'delaware', 'south dakota', 'north dakota', 'alaska', 'vermont',
            'wyoming'
        ]

        # Check if any non-Alabama state is mentioned
        for state in non_alabama_states:
            if state in text_lower:
                return {
                    'valid_location': False,
                    'error_message': "This app is currently only configured for the state of Alabama in the USA",
                    'county': None,
                    'exact_match': False
                }

        # Check for explicit country that's not USA
        non_usa_countries = ['canada', 'mexico', 'britain', 'england', 'france', 'germany', 'china', 'japan']
        for country in non_usa_countries:
            if country in text_lower and not any(keyword in text_lower for keyword in country_keywords):
                return {
                    'valid_location': False,
                    'error_message': "This app is currently only configured for the state of Alabama in the USA",
                    'county': None,
                    'exact_match': False
                }

        # If no Alabama or USA mentioned, but other location indicators present
        location_indicators = ['county', 'state', 'country', 'city', 'in ', 'at ', 'from ']
        has_location_indicator = any(indicator in text_lower for indicator in location_indicators)

        # If location indicators are present but no Alabama/USA mentioned, check for county match first
        county_match_result = self._find_county_match(text)

        if has_location_indicator and not any(keyword in text_lower for keyword in state_keywords + country_keywords):
            if not county_match_result['county']:
                return {
                    'valid_location': False,
                    'error_message': "This app is currently only configured for the state of Alabama in the USA",
                    'county': None,
                    'exact_match': False
                }

        # If we get here, either Alabama/USA was mentioned or we found a county match
        return county_match_result

    def _find_county_match(self, text: str) -> Dict:
        """Find the best matching Alabama county"""
        logger.info(f"Finding county match in: {text}")

        text_lower = text.lower()

        # First try exact matches (case insensitive)
        for county in self.alabama_counties:
            county_lower = county.lower()
            # Check for exact county name match
            if county_lower in text_lower:
                # Make sure it's a word boundary match
                county_pattern = r'\b' + re.escape(county_lower) + r'\b'
                if re.search(county_pattern, text_lower):
                    logger.info(f"Found exact county match: {county}")
                    return {
                        'valid_location': True,
                        'error_message': None,
                        'county': county,
                        'exact_match': True
                    }

        # If no exact match, try fuzzy matching
        # Extract potential county names (words that might be counties)
        words = re.findall(r'\b[A-Za-z]+\b', text)

        best_match = None
        best_score = 0

        for word in words:
            # Skip common words that aren't counties
            if word.lower() in ['county', 'alabama', 'state', 'usa', 'america', 'united', 'states']:
                continue

            matches = get_close_matches(word, self.alabama_counties, n=1, cutoff=0.6)
            if matches:
                # Calculate a simple similarity score
                match = matches[0]
                score = len(set(word.lower()) & set(match.lower())) / len(set(word.lower()) | set(match.lower()))
                if score > best_score:
                    best_match = match
                    best_score = score

        if best_match:
            logger.info(f"Found fuzzy county match: {best_match}")
            return {
                'valid_location': True,
                'error_message': f"Did you mean {best_match}?",
                'county': best_match,
                'exact_match': False
            }

        # No match found
        logger.warning("No county match found")
        return {
            'valid_location': True,  # Still valid for Alabama, just no specific county
            'error_message': "There was no matching county found in Alabama",
            'county': None,
            'exact_match': False
        }

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
            model=config["model_name"],
            verbose=True
        )
        self.parser = ExperimentParser(self.llm)

        # Build the workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ExperimentState)

        # Add nodes
        workflow.add_node("improve_query", self._improve_query_node)
        ###### CHANGES
        workflow.add_node("detect_intent", self._detect_intent_node)
        workflow.add_node("provide_capabilities", self._provide_capabilities_node)
        #### END_CHANGES
        workflow.add_node("parse_input", self._parse_input_node)
        workflow.add_node("check_completeness", self._check_completeness_node)
        workflow.add_node("ask_clarification", self._ask_clarification_node)
        workflow.add_node("run_experiment", self._run_experiment_node)
        workflow.add_node("generate_output", self._generate_output_node)
        workflow.add_node("end_conversation", self._end_conversation_node)

        # Define the flow
        workflow.set_entry_point("improve_query")

        ###### CHANGES
        workflow.add_edge("improve_query", "detect_intent")

        workflow.add_conditional_edges(
            "detect_intent",
            self._should_provide_capabilities,
            {
                "capabilities": "provide_capabilities",
                "experiment": "parse_input"
            }
        )

        workflow.add_edge("provide_capabilities", END)
        #### END_CHANGES

        workflow.add_edge("parse_input", "check_completeness")

        workflow.add_conditional_edges(
            "check_completeness",
            self._should_ask_clarification,
            {
                "clarify": "ask_clarification",
                "proceed": "run_experiment",
                "end": "end_conversation"
            }
        )

        workflow.add_edge("ask_clarification", END)
        workflow.add_edge("run_experiment", "generate_output")
        workflow.add_edge("generate_output", END)
        workflow.add_edge("end_conversation", END)

        return workflow.compile(checkpointer=MemorySaver())

    def _improve_query_node(self, state: ExperimentState) -> Dict:
        """Improve the user query by analyzing conversation context and extracting key information"""
        logger.info("=== IMPROVE QUERY NODE ===")
        logger.info(f"Original query: {state.user_query}")

        # Check if the query contains conversation context
        if "User:" in state.user_query and "Assistant:" in state.user_query:
            # Extract conversation parts
            parts = state.user_query.split("User:")

            # Get the most recent user message
            if len(parts) > 1:
                latest_user_msg = parts[-1].strip()
                # Remove any trailing "Assistant:" content
                if "Assistant:" in latest_user_msg:
                    latest_user_msg = latest_user_msg.split("Assistant:")[0].strip()
            else:
                latest_user_msg = state.user_query.strip()

            ###### CHANGES
            # Extract context from previous messages with better parameter tracking
            conversation_context = ""
            all_user_messages = []

            if len(parts) > 2:  # There's prior conversation
                # Collect all user messages to track parameter evolution
                for i in range(1, len(parts)):
                    user_msg = parts[i].strip()
                    if "Assistant:" in user_msg:
                        user_msg = user_msg.split("Assistant:")[0].strip()
                    if user_msg:
                        all_user_messages.append(user_msg)

                conversation_context = "\n".join([f"Previous message {i + 1}: {msg}"
                                                  for i, msg in enumerate(all_user_messages[:-1])])

            # Create improvement prompt with better parameter prioritization
            improvement_prompt = f"""
            Analyze this conversation and improve the latest user message by ONLY incorporating information that was explicitly mentioned in previous messages.

            Conversation history:
            {conversation_context if conversation_context else "No previous context"}

            Latest user message:
            {latest_user_msg}

            CRITICAL RULES:
            1. ONLY use information that was explicitly stated in the conversation history
            2. DO NOT add any assumptions, defaults, or information not mentioned
            3. If no previous context exists, return the latest message unchanged
            4. If previous context exists, only add information that was clearly stated
            5. Keep the natural language format

            Task: Create an improved query that combines the latest message with ONLY the explicitly mentioned information from previous messages.

            If the conversation history contains no relevant agricultural parameters (location, dates, fertilization, season), return the latest user message exactly as is.

            Return only the improved query:
            """
            #### END_CHANGES

            try:
                logger.info("Sending query improvement prompt to LLM")
                improved_query = self.llm.invoke(improvement_prompt).strip()
                logger.info(f"Improved query: {improved_query}")

                return {"user_query": improved_query}
            except Exception as e:
                logger.error(f"Error improving query: {e}")
                # Fallback to using just the latest user message
                return {"user_query": latest_user_msg}
        else:
            # No conversation context, use original query
            logger.info("No conversation context detected, using original query")
            return {"user_query": state.user_query}


    ###### CHANGES
    def _detect_intent_node(self, state: ExperimentState) -> Dict:
        """Detect user intent to determine if they want to run an experiment or just learn about capabilities"""
        logger.info("=== DETECT INTENT NODE ===")

        intent_prompt = f"""
        Analyze the following user query to determine their intent:

        Query: "{state.user_query}"

        Determine if the user is:
        1. Asking about what this system can do / its capabilities
        2. Making a general inquiry about agricultural modeling
        3. Requesting to run a specific agricultural experiment (even if incomplete)

        Keywords that indicate experiment intent:
        - "run", "test", "simulate", "model", "experiment", "evaluate"
        - Mentions specific parameters like dates, locations, fertilization rates
        - "what would happen if", "predict", "estimate yield"

        Keywords that indicate capability inquiry:
        - "what can you do", "how does this work", "what is this", "help"
        - "capabilities", "features", "about"
        - General questions without specific parameters

        Respond with only:
        "EXPERIMENT" if they want to run an experiment
        "CAPABILITIES" if they want to learn about capabilities
        """

        try:
            intent_response = self.llm.invoke(intent_prompt).strip().upper()
            logger.info(f"Detected intent: {intent_response}")

            is_experiment_intent = "EXPERIMENT" in intent_response

            return {"experiment_intent": is_experiment_intent}

        except Exception as e:
            logger.error(f"Error detecting intent: {e}")
            # Default to experiment intent to maintain existing behavior
            return {"experiment_intent": True}

    def _should_provide_capabilities(self, state: ExperimentState) -> str:
        """Determine if we should provide capabilities or proceed with experiment"""
        return "experiment" if getattr(state, 'experiment_intent', True) else "capabilities"

    def _provide_capabilities_node(self, state: ExperimentState) -> Dict:
        """Provide information about system capabilities in a natural way"""
        logger.info("=== PROVIDE CAPABILITIES NODE ===")

        capabilities_prompt = f"""
        The user asked: "{state.user_query}"

        You are an agricultural analysis agent that can predict crop yields. Generate a brief, friendly response from YOUR perspective (first person) that:

        1. Introduces yourself simply
        2. Mentions you can predict yields 
        3. Asks if they'd like to set up an experiment

        Keep it SHORT and conversational - no more than 2-3 sentences. Examples of good responses:
        "Hi! I'm an agricultural analysis agent that can predict crop yields for Alabama farms. Would you like to set up an experiment?"
        "That's great! I can help predict yields for different planting scenarios in Alabama. Would you like to run an experiment?"

        Write from YOUR perspective as the agent, keep it brief and friendly:
        """

        try:
            response = self.llm.invoke(capabilities_prompt).strip()
            logger.info(f"Generated capabilities response: {response}")

            return {"natural_language_output": response}

        except Exception as e:
            logger.error(f"Error generating capabilities response: {e}")
            default_response = "Hi! I'm an agricultural analysis agent that can predict crop yields for Alabama farms. Would you like to set up an experiment?"

            return {"natural_language_output": default_response}

    #### END_CHANGES

    def _parse_input_node(self, state: ExperimentState) -> Dict:
        """Parse the user input to extract parameters"""
        logger.info("=== PARSE INPUT NODE ===")
        logger.info(f"Processing query: {state.user_query}")

        text = state.user_query

        # Parse all components
        location_result = self.parser.parse_location(text)
        planting_date = self.parser.parse_date(text)
        fertilization_plan = self.parser.parse_fertilization_plan(text)
        planting_length = self.parser.parse_planting_length(text)

        # Convert planting length to cultivar
        cultivar = None
        if planting_length:
            cultivar = CULTIVAR_MAPPING.get(planting_length)

        logger.info(
            f"Parsed - Date: {planting_date}, Fert: {fertilization_plan}, Length: {planting_length}, Cultivar: {cultivar}, County: {location_result['county']}")

        return {
            "planting_date": planting_date,
            "fertilization_plan": fertilization_plan,
            "planting_length": planting_length,
            "cultivar": cultivar,
            "county": location_result['county'],
            "location_exact_match": location_result['exact_match'],
            "location_valid": location_result['valid_location'],
            "location_error_message": location_result['error_message']
        }

    def _check_completeness_node(self, state: ExperimentState) -> Dict:
        """Check if all required information is present"""
        logger.info("=== CHECK COMPLETENESS NODE ===")

        missing = []

        # Check location validity and completeness
        if not state.location_valid:
            # Don't add to missing info - we'll end the conversation
            logger.info("Invalid location detected - will end conversation")
            return {
                "missing_info": [],
                "needs_clarification": False,
                "should_end_conversation": True
            }
        elif state.location_error_message:
            if "Did you mean" in state.location_error_message:
                missing.append("location confirmation")
            elif "no matching county" in state.location_error_message:
                missing.append("specific Alabama county")

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
            "needs_clarification": needs_clarification,
            "should_end_conversation": False
        }

    def _should_ask_clarification(self, state: ExperimentState) -> str:
        """Determine if clarification is needed"""
        # Check if we should end the conversation due to invalid location
        if hasattr(state, 'should_end_conversation') and state.should_end_conversation:
            return "end"
        return "clarify" if state.needs_clarification else "proceed"

    def _end_conversation_node(self, state: ExperimentState) -> Dict:
        """End conversation with appropriate message for invalid location"""
        logger.info("=== END CONVERSATION NODE ===")

        if not state.location_valid and state.location_error_message:
            end_message = state.location_error_message
        else:
            end_message = "This app is currently only configured for the state of Alabama in the USA"

        logger.info(f"Ending conversation: {end_message}")

        return {
            "natural_language_output": end_message
        }

    def _ask_clarification_node(self, state: ExperimentState) -> Dict:
        """Ask for clarification on missing information"""
        logger.info("=== ASK CLARIFICATION NODE ===")

        if not state.missing_info:
            return {"user_query": state.user_query}

        ###### CHANGES
        # Create a more natural and concise clarification request
        clarification_prompt = f"""
        You are an agricultural analysis agent. The user wants to run an experiment but some information is missing: {', '.join(state.missing_info)}

        Current information we have:
        - Location: {state.county if state.county else 'Not specified'}
        - Planting date: {state.planting_date if state.planting_date else 'Not specified'}
        - Fertilization plan: {state.fertilization_plan if state.fertilization_plan else 'Not specified'}
        - Season length: {state.planting_length if state.planting_length else 'Not specified'}

        Generate a SHORT, friendly request asking for the missing information. Be conversational and helpful, but keep it brief (2-3 sentences max).

        Write from YOUR perspective as the agent. Examples:
        "I need a few more details to run your experiment. What's your planting date and which Alabama county?"
        "Great! I just need to know when you're planting and your fertilization plan to get started."

        Keep it simple and direct:
        """

        try:
            response = self.llm.invoke(clarification_prompt).strip()
            logger.info(f"Generated clarification request: {response}")

            return {"natural_language_output": response}

        except Exception as e:
            logger.error(f"Error generating clarification: {e}")

            # Fallback to original logic
            clarification_parts = []
            examples = []

            for missing_item in state.missing_info:
                if missing_item == "location confirmation":
                    clarification_parts.append(f"Location: {state.location_error_message}")
                    examples.append(
                        "For location confirmation: Please confirm the county name or provide a different Alabama county")
                elif missing_item == "specific Alabama county":
                    clarification_parts.append(f"Location: {state.location_error_message}")
                    examples.append(
                        "For location: Please specify an Alabama county (e.g., \"Mobile County\" or \"Jefferson County\")")
                elif "planting date" in missing_item:
                    clarification_parts.append("planting date")
                    examples.append("For planting date: \"2024-03-15\" or \"March 15, 2024\"")
                elif "fertilization plan" in missing_item:
                    clarification_parts.append("fertilization plan")
                    examples.append(
                        "For fertilization plan: \"100 kg/ha nitrogen at planting, 50 kg/ha at 30 days after planting\"")
                elif "planting length" in missing_item or "season" in missing_item:
                    clarification_parts.append("planting length/season")
                    examples.append("For planting length: \"short season\" or \"medium season\" or \"long season\"")
                else:
                    clarification_parts.append(missing_item)

            missing_str = "\n- ".join(clarification_parts)
            examples_str = "\n".join(examples) if examples else ""

            clarification_message = f"""
        I need more information to run the experiment. The following information is missing or unclear:

        - {missing_str}

        Please provide the missing information. Here are some examples:

        {examples_str}
        """

            return {"natural_language_output": clarification_message}
        #### END_CHANGES

    def _run_experiment_node(self, state: ExperimentState) -> Dict:
        """Run the agricultural experiment"""
        logger.info("=== RUN EXPERIMENT NODE ===")

        try:
            print("INVOKING EXPERIMENT")
            print("COUNTY IS " + state.county)
            results = experiment_tool.invoke({
                "planting_date": state.planting_date,
                "fert_plan": state.fertilization_plan,
                "cultivar": state.cultivar,
                "county": state.county
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

        # Format fertilization plan for display
        fert_display = []
        if state.fertilization_plan:
            for rate, days in state.fertilization_plan:
                timing = "at planting" if days == 0 else f"{int(days)} days after planting"
                fert_display.append(f"{rate} kg/ha nitrogen {timing}")
        fert_plan_str = ", ".join(fert_display) if fert_display else "No fertilization plan specified"

        ###### CHANGES
        # Create a more natural and engaging prompt for output generation
        prompt = f"""
        Generate a natural, conversational summary of the agricultural experiment results. Make it sound like you're talking to a farmer or agricultural professional.

        Experiment Parameters:
        - Planting Date: {state.planting_date}
        - Cultivar: {state.cultivar} ({state.planting_length} season)
        - Fertilization Plan: {fert_plan_str}
        - Location: {state.county or 'Alabama'}

        Results:
        Yield Results: {results.get('yield_results', {})}
        Water Stress Results: {results.get('water_stress', {})}
        Nitrogen Stress Results: {results.get('nitro_stress', {})}

        Structure your response as follows:
        1. Brief confirmation of what was modeled
        2. Key yield predictions with confidence intervals (make this the highlight)
        3. Summary of any stress factors that affected growth
        4. End with a natural offer to explore different scenarios - vary this each time (examples: "Would you like to see how different planting dates might affect yield?", "I can also model what happens with different fertilization strategies if you're interested", "Feel free to ask about other scenarios or parameter combinations")

        Make it conversational and helpful, not technical or robotic. Focus on actionable insights for agricultural decision-making.
        Only discuss nitrogen applications as specified in the fertilization plan - don't mention other nutrients unless explicitly provided.
        """
        #### END_CHANGES

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

    def verbose_evaluate(self, user_query: str) -> Tuple[str, Dict]:
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
        return result, final_state_dict

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
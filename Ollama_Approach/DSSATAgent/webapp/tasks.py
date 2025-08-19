import json
import time
import asyncio
from asgiref.sync import sync_to_async
from celery import shared_task
from django.utils import timezone
from .models import Chat, Message, Chart
from .agent.economic_evaluator import EconomicEvaluatorAgent, ExperimentState
from .plot import format_agricultural_charts


# Mock LangGraph agent - replace with your actual LangGraph implementation
class MockLangGraphAgent:
    """
    Mock LangGraph agent for demonstration.
    Replace this with your actual LangGraph agent implementation.
    """

    def __init__(self, chat_id, agent_state=None):
        self.chat_id = chat_id
        self.state = agent_state or {}

    def process_message(self, message_content):
        """
        Process a message and return response with optional charts.
        Replace this with your actual LangGraph workflow.
        """
        # Simulate processing time
        time.sleep(2)

        # Update agent state (example)
        if 'message_count' not in self.state:
            self.state['message_count'] = 0
        self.state['message_count'] += 1

        # Generate mock response
        response_content = f"This is a mock response to: '{message_content}'. Message count: {self.state['message_count']}"

        # Mock chart data (conditionally generate charts)
        charts_data = []
        if 'chart' in message_content.lower() or 'graph' in message_content.lower():
            charts_data = [
                {
                    'type': 'line',
                    'title': 'Sample Line Chart',
                    'data': {
                        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
                        'datasets': [{
                            'label': 'Sample Data',
                            'data': [10, 25, 15, 40, 30],
                            'borderColor': 'rgb(75, 192, 192)',
                            'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                        }]
                    }
                },
                {
                    'type': 'bar',
                    'title': 'Sample Bar Chart',
                    'data': {
                        'labels': ['A', 'B', 'C', 'D'],
                        'datasets': [{
                            'label': 'Sample Values',
                            'data': [12, 19, 3, 17],
                            'backgroundColor': [
                                'rgba(255, 99, 132, 0.2)',
                                'rgba(54, 162, 235, 0.2)',
                                'rgba(255, 205, 86, 0.2)',
                                'rgba(75, 192, 192, 0.2)'
                            ],
                            'borderColor': [
                                'rgba(255, 99, 132, 1)',
                                'rgba(54, 162, 235, 1)',
                                'rgba(255, 205, 86, 1)',
                                'rgba(75, 192, 192, 1)'
                            ],
                            'borderWidth': 1
                        }]
                    }
                }
            ]

        return response_content, charts_data, self.state


@shared_task(bind=True)
def process_message_task(self, chat_id, message_id, message_content):
    """
    Modified Celery task using astream_events for real-time node tracking
    """
    # Run the async workflow in the Celery task
    final_state =  asyncio.run(
        _async_process_message(self, chat_id, message_id, message_content)
    )

    return final_state


async def _async_process_message(task_self, chat_id, message_id, message_content):
    agent = EconomicEvaluatorAgent()
    initial_state = ExperimentState(user_query=message_content)
    config_dict = {"configurable": {"thread_id": chat_id}}

    # Convert sync Django ORM calls to async
    chat = await sync_to_async(Chat.objects.get)(id=chat_id)
    message = await sync_to_async(Message.objects.get)(id=message_id)

    message.status = 'pending'
    await sync_to_async(message.save)(update_fields=['status'])

    final_state_dict = {}
    current_node_output = {}

    try:
        # Use astream_events for granular event tracking
        async for event in agent.workflow.astream_events(
                initial_state,
                config_dict,
                version="v1"
        ):
            event_type = event["event"]

            # Handle node start events
            if event_type == "on_chain_start":
                metadata = event.get("metadata", {})
                if "langgraph_node" in metadata:
                    node_name = metadata["langgraph_node"]
                    print(f"Starting node: {node_name}")

                    # Update current node immediately when it starts
                    message.current_node = node_name
                    await sync_to_async(message.save)(update_fields=['current_node'])

            # Handle node completion events
            elif event_type == "on_chain_end":
                metadata = event.get("metadata", {})
                if "langgraph_node" in metadata:
                    node_name = metadata["langgraph_node"]
                    node_output = event.get("data", {}).get("output")

                    print(f"Completed node: {node_name}")

                    # Store the node output
                    if isinstance(node_output, dict):
                        current_node_output[node_name] = node_output
                        final_state_dict.update(node_output)

            # Handle workflow completion
            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                # This indicates the entire workflow has completed
                final_output = event.get("data", {}).get("output", {})
                if isinstance(final_output, dict):
                    final_state_dict.update(final_output)

        # Task finished successfully
        message.content = final_state_dict.get("natural_language_output", "")
        message.status = 'completed'
        message.current_node = None
        await sync_to_async(message.save)(update_fields=['content', 'status', 'current_node'])

        chat.agent_state = final_state_dict  # Use final_state_dict instead of state
        chat.is_processing = False
        chat.updated_at = timezone.now()
        await sync_to_async(chat.save)()

        charts_data = final_state_dict.get('experiment_results', {})

        if charts_data:
            formatted_charts = format_agricultural_charts(charts_data)
        else:
            formatted_charts = []

        for chart_data in formatted_charts:
            chart = await sync_to_async(Chart.objects.create)(
                message=message,
                chart_type=chart_data['type'],
                title=chart_data['title'],
                data=chart_data['data']
            )

        return final_state_dict

    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
        message.status = 'failed'
        message.current_node = None
        await sync_to_async(message.save)(update_fields=['status', 'current_node'])

        chat.is_processing = False
        await sync_to_async(chat.save)(update_fields=['is_processing'])
        raise e
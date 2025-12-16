import json
import time
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from .models import Chat, Message, Chart
from .tasks import process_message_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_time(t):
    minutes = int(t // 60)
    seconds = int(t % 60)
    milliseconds = int((t * 1000) % 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

class ChatView(View):
    """View for the chat interface"""

    def get(self, request, chat_id):
        """Render the chat page"""
        start = time.perf_counter()
        chat = get_object_or_404(Chat, id=chat_id)
        messages = chat.messages.all().order_by('created_at')

        # Serialize messages for the template
        messages_data = []
        for msg in messages:
            message_data = {
                'id': str(msg.id),
                'type': msg.message_type,
                'content': msg.content,
                'created_at': msg.created_at.isoformat(),
                'status': msg.status,
                'charts': []
            }

            # Add charts if they exist
            for chart in msg.charts.all():
                message_data['charts'].append({
                    'id': str(chart.id),
                    'type': chart.chart_type,
                    'title': chart.title,
                    'data': chart.data
                })

            messages_data.append(message_data)

        context = {
            'chat': chat,
            'messages': messages_data,
            'chat_id': str(chat.id),
            'is_processing': chat.is_processing
        }
        end = time.perf_counter()
        logger.info("Pre-render processing time (ChatView): "+ str(format_time(end - start)))
        return render(request, 'webapp/chat.html', context)


@method_decorator(csrf_exempt, name='dispatch')
class MessageAPIView(View):
    """API endpoint for sending new messages"""

    def post(self, request, chat_id):
        """Send a new message"""
        start = time.perf_counter()
        try:
            data = json.loads(request.body)
            content = data.get('content', '').strip()

            if not content:
                return JsonResponse({'error': 'Message content is required'}, status=400)

            chat = get_object_or_404(Chat, id=chat_id)

            # Build conversation history in the format expected by the agent
            conversation_parts = []
            messages = chat.messages.filter(
                message_type__in=['user', 'assistant']
            ).order_by('created_at')

            for message in messages:
                if message.message_type == 'user':
                    conversation_parts.append(f"User: {message.content}")
                elif message.message_type == 'assistant' and message.content.strip():
                    # Only include completed assistant messages with content
                    conversation_parts.append(f"Assistant: {message.content}")

            # Add the current user message to the conversation
            conversation_parts.append(f"User: {content}")

            # Compile the full conversation context
            if len(conversation_parts) > 1:  # More than just the current message
                compiled_conversation = "\n\n".join(conversation_parts)
            else:
                compiled_conversation = content  # First message, no history needed

            # Create user message
            user_message = Message.objects.create(
                chat=chat,
                message_type='user',
                content=content,
                status='completed'
            )

            # Create pending assistant message
            assistant_message = Message.objects.create(
                chat=chat,
                message_type='assistant',
                content='',
                status='pending'
            )

            # Mark chat as processing
            chat.is_processing = True
            chat.save()

            # Send the compiled conversation to the agent
            task = process_message_task.delay(
                str(chat.id),
                str(assistant_message.id),
                compiled_conversation  # This now includes the full conversation context
            )

            # Store task ID for tracking
            assistant_message.task_id = task.id
            assistant_message.save()

            end = time.perf_counter()
            logger.info("Response processing time (MessageAPIView): "+ str(format_time(end - start)))
            return JsonResponse({
                'success': True,
                'user_message_id': str(user_message.id),
                'assistant_message_id': str(assistant_message.id),
                'task_id': task.id
            })

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class MessageStatusAPIView(View):
    """API endpoint for checking message processing status"""

    def get(self, request, chat_id, message_id):
        """Check if message processing is complete"""
        start = time.perf_counter()
        try:
            chat = get_object_or_404(Chat, id=chat_id)
            message = get_object_or_404(Message, id=message_id, chat=chat)

            response_data = {
                'status': message.status,
                'is_processing': chat.is_processing,
                'current_node': message.current_node,  # new
            }

            if message.status == 'completed':
                # Include message content and any charts
                response_data.update({
                    'content': message.content,
                    'charts': []
                })

                # Add chart data
                for chart in message.charts.all():
                    response_data['charts'].append({
                        'id': str(chart.id),
                        'type': chart.chart_type,
                        'title': chart.title,
                        'data': chart.data
                    })

            elif message.status == 'failed':
                response_data['error'] = 'Message processing failed'
            end = time.perf_counter()
            logger.info("Response processing time (MessageStatusAPIView): "+ str(format_time(end - start)))

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class NewChatView(View):
    """View to create a new chat and redirect to it"""

    def get(self, request):
        """Create a new chat and redirect to the chat page"""
        start = time.perf_counter()
        # Create a new chat with a default title
        chat = Chat.objects.create(title='New Chat')

        # Create assistant message
        assistant_message = Message.objects.create(
            chat=chat,
            message_type='assistant',
            content='Hello! I am the DSSATAgent, a tool from Earth RISE that can help you predict farm yield using'
                    ' a well-studied agricultural model. Please provide your county, planting date, fertilization plan,'
                    ' and crop variation. Right now, I can only assist with experiments in Alabama using maize.',
            status='completed'
        )

        end = time.perf_counter()
        logger.info("Response processing time (NewChatView:Get): "+ str(format_time(end - start)))

        # Redirect to the chat page
        from django.shortcuts import redirect
        return redirect('webapp:chat', chat_id=chat.id)

    def post(self, request):
        """Create a new chat with optional title and redirect"""
        start = time.perf_counter()
        title = request.POST.get('title', 'New Chat')
        chat = Chat.objects.create(title=title)
        mid_start = time.perf_counter()

        # Create assistant message
        assistant_message = Message.objects.create(
            chat=chat,
            message_type='assistant',
            content='Hello! I am the DSSATAgent, a tool from Earth RISE that can help you predict farm yield using'
                    ' a well-studied agricultural model. Please provide your county, planting date, fertilization plan,'
                    ' and crop variation. Right now, I can only assist with experiments in Alabama using maize.',
            status='completed'
        )

        end = time.perf_counter()
        logger.info("Response processing time (NewChatView:Post): "+ str(format_time(end - start)))

        from django.shortcuts import redirect
        return redirect('webapp:chat', chat_id=chat.id)


@method_decorator(csrf_exempt, name='dispatch')
class NewChatAPIView(View):
    """API endpoint for creating new chats"""

    def post(self, request):
        """Create a new chat"""
        start = time.perf_counter()
        try:
            data = json.loads(request.body) if request.body else {}
            title = data.get('title', '')

            chat = Chat.objects.create(title=title)

            # Create assistant message
            assistant_message = Message.objects.create(
                chat=chat,
                message_type='assistant',
                content='Hello! I am the DSSATAgent, a tool from Earth RISE that can help you predict farm yield using'
                        ' a well-studied agricultural model. Please provide your county, planting date, fertilization plan,'
                        ' and crop variation. Right now, I can only assist with experiments in Alabama using maize.',
                status='completed'
            )

            end = time.perf_counter()
            logger.info("Response processing time (NewChatAPIView): "+ str(format_time(end - start)))

            return JsonResponse({
                'success': True,
                'chat_id': str(chat.id),
                'redirect_url': f'/chats/{chat.id}/'
            })

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


# Add to webapp/views.py

class HomeView(View):
    """Optional home page view"""

    def get(self, request):
        """Show home page with recent chats"""
        start = time.perf_counter()
        # Get recent chats (last 5)
        recent_chats = Chat.objects.all()[:5]

        context = {
            'recent_chats': recent_chats
        }

        end = time.perf_counter()
        logger.info("Response processing time (HomeView): "+ str(format_time(end - start)))

        return render(request, 'webapp/home.html', context)


class TestView(View):
    def post(self, request):
        return JsonResponse({'hi': 'hi'})
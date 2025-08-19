import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from .models import Chat, Message, Chart
from .tasks import process_message_task


class ChatView(View):
    """View for the chat interface"""

    def get(self, request, chat_id):
        """Render the chat page"""
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
        return render(request, 'webapp/chat.html', context)


@method_decorator(csrf_exempt, name='dispatch')
class MessageAPIView(View):
    """API endpoint for sending new messages"""

    def post(self, request, chat_id):
        """Send a new message"""
        try:
            data = json.loads(request.body)
            content = data.get('content', '').strip()

            if not content:
                return JsonResponse({'error': 'Message content is required'}, status=400)

            chat = get_object_or_404(Chat, id=chat_id)

            compiled_messages = "\n".join(
                chat.messages
                .filter(message_type='user')
                .order_by('created_at')
                .values_list('content', flat=True)
            )

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

            content = content + ' ' + compiled_messages

            # Start async processing
            task = process_message_task.delay(
                str(chat.id),
                str(assistant_message.id),
                content
            )

            # Store task ID for tracking
            assistant_message.task_id = task.id
            assistant_message.save()

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

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


class NewChatView(View):
    """View to create a new chat and redirect to it"""

    def get(self, request):
        """Create a new chat and redirect to the chat page"""
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

        # Redirect to the chat page
        from django.shortcuts import redirect
        return redirect('webapp:chat', chat_id=chat.id)

    def post(self, request):
        """Create a new chat with optional title and redirect"""
        title = request.POST.get('title', 'New Chat')
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

        from django.shortcuts import redirect
        return redirect('webapp:chat', chat_id=chat.id)


@method_decorator(csrf_exempt, name='dispatch')
class NewChatAPIView(View):
    """API endpoint for creating new chats"""

    def post(self, request):
        """Create a new chat"""
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
        # Get recent chats (last 5)
        recent_chats = Chat.objects.all()[:5]

        context = {
            'recent_chats': recent_chats
        }
        return render(request, 'webapp/home.html', context)


class TestView(View):
    def post(self, request):
        return JsonResponse({'hi': 'hi'})
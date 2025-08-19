import uuid
import json
from django.db import models
from django.utils import timezone


class Chat(models.Model):
    """Model to store chat sessions"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    title = models.CharField(max_length=200, blank=True, null=True)

    # Store LangGraph agent state as JSON
    agent_state = models.JSONField(default=dict, blank=True)

    # Track if chat is currently processing
    is_processing = models.BooleanField(default=False)

    def __str__(self):
        return f"Chat {self.id} - {self.title or 'Untitled'}"

    class Meta:
        ordering = ['-updated_at']


class Message(models.Model):
    """Model to store individual messages in a chat"""
    MESSAGE_TYPES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]

    PROCESSING_STATUS = [
        ('completed', 'Completed'),
        ('pending', 'Pending'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    # For tracking message processing
    status = models.CharField(max_length=10, choices=PROCESSING_STATUS, default='completed')
    task_id = models.CharField(max_length=255, blank=True, null=True)  # Celery task ID
    current_node = models.CharField(max_length=255, blank=True, null=True)  # new

    # Store any charts/attachments as JSON
    charts_data = models.JSONField(default=list, blank=True)

    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."

    class Meta:
        ordering = ['created_at']


class Chart(models.Model):
    """Model to store chart data for messages"""
    CHART_TYPES = [
        ('line', 'Line Chart'),
        ('bar', 'Bar Chart'),
        ('scatter', 'Scatter Plot'),
        ('pie', 'Pie Chart'),
        ('heatmap', 'Heatmap'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='charts')
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES)
    title = models.CharField(max_length=200)
    data = models.JSONField()  # Store chart data/config
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.chart_type}: {self.title}"

    class Meta:
        ordering = ['created_at']
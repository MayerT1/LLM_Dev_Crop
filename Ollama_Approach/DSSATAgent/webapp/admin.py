from django.contrib import admin
from django.utils.html import format_html
from .models import Chat, Message, Chart


@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ('id', 'title', 'message_count', 'is_processing', 'created_at', 'updated_at')
    list_filter = ('is_processing', 'created_at', 'updated_at')
    search_fields = ('title', 'id')
    readonly_fields = ('id', 'created_at', 'updated_at', 'message_count')

    def message_count(self, obj):
        return obj.messages.count()

    message_count.short_description = 'Messages'

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'title', 'created_at', 'updated_at', 'message_count')
        }),
        ('State', {
            'fields': ('is_processing', 'agent_state'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'chat_link', 'message_type', 'content_preview', 'status', 'chart_count', 'created_at')
    list_filter = ('message_type', 'status', 'created_at')
    search_fields = ('content', 'chat__title', 'chat__id')
    readonly_fields = ('id', 'created_at', 'chart_count')
    raw_id_fields = ('chat',)

    def content_preview(self, obj):
        return obj.content[:100] + '...' if len(obj.content) > 100 else obj.content

    content_preview.short_description = 'Content'

    def chat_link(self, obj):
        url = f'/admin/webapp/chat/{obj.chat.id}/change/'
        return format_html('<a href="{}">{}</a>', url, str(obj.chat.id)[:8])

    chat_link.short_description = 'Chat'

    def chart_count(self, obj):
        return obj.charts.count()

    chart_count.short_description = 'Charts'

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'chat', 'message_type', 'created_at', 'chart_count')
        }),
        ('Content', {
            'fields': ('content', 'status', 'task_id')
        }),
        ('Charts Data', {
            'fields': ('charts_data',),
            'classes': ('collapse',)
        }),
    )


@admin.register(Chart)
class ChartAdmin(admin.ModelAdmin):
    list_display = ('id', 'message_link', 'chart_type', 'title', 'created_at')
    list_filter = ('chart_type', 'created_at')
    search_fields = ('title', 'message__content')
    readonly_fields = ('id', 'created_at')
    raw_id_fields = ('message',)

    def message_link(self, obj):
        url = f'/admin/webapp/message/{obj.message.id}/change/'
        return format_html('<a href="{}">{}</a>', url, str(obj.message.id)[:8])

    message_link.short_description = 'Message'

    fieldsets = (
        ('Basic Information', {
            'fields': ('id', 'message', 'chart_type', 'title', 'created_at')
        }),
        ('Chart Data', {
            'fields': ('data',),
            'classes': ('collapse',)
        }),
    )
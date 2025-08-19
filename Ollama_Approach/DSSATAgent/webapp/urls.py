from django.urls import path
from . import views

app_name = 'webapp'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),

    # New chat creation endpoint
    path('chat/', views.NewChatView.as_view(), name='new_chat'),

    # Chat interface
    path('chats/<uuid:chat_id>/', views.ChatView.as_view(), name='chat'),

    # API endpoints
    path('api/chats/new/', views.NewChatAPIView.as_view(), name='new_chat_api'),
    path('api/chats/<uuid:chat_id>/messages/', views.MessageAPIView.as_view(), name='message_api'),
    path('api/chats/<uuid:chat_id>/messages/<uuid:message_id>/status/', views.MessageStatusAPIView.as_view(),
         name='message_status_api'),
    path('api/test/', views.TestView.as_view(), name='test')
]
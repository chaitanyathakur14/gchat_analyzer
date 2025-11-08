from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('reset_csv/', views.reset_csv, name='reset_csv'), 
    path('sentiment/', views.sentiment_view, name='sentiment'),
    path('topics/', views.topics_view, name='topics'),
    path('quality/', views.quality_view, name='quality'),
    path("api/metrics/", views.api_metrics, name="api_metrics"),
    path("api/activity/", views.api_activity, name="api_activity"), 
     path("api/messages/", views.api_messages, name="api_messages"),
     path("api/topics/", views.api_topics, name="api_topics"),
    path("api/sentiment-summary/", views.api_sentiment_summary, name="api_sentiment_summary"),
    path("api/conversations/", views.api_conversations, name="api_conversations"),
    path('repetition/', views.repetition_view, name='repetition'),




]

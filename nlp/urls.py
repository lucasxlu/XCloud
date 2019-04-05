from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    url('wordseg', views.word_seg, name='wordseg'),
    url('sentimentview', views.sentiment_view, name='sentimentview'),
    url('sentiment', views.sentiment, name='sentiment'),
]

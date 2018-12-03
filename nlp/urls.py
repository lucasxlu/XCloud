from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    url('wordseg', views.word_seg, name='wordseg'),
    url('sentiment', views.sentiment, name='sentiment'),
]

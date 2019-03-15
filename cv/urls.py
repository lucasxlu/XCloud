from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    path('index', views.index, name='index'),
    path('mcloud', views.mcloud, name='mcloud'),
    path('fbp', views.fbp, name='fbp'),
    url('fbpview', views.fbp_view, name='fbpview'),
    url('detectface', views.detect_face, name='detectface'),
]

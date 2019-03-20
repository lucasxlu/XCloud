from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    path('index', views.index, name='index'),
    path('mcloud', views.mcloud, name='mcloud'),
    # path('mcloud/skin', views.rec_skin, name='recskin'),
    # path('sdr', views.sdr, name='sdr'),
    url('fbpview', views.fbp_view, name='fbpview'),
    url('fbp', views.fbp, name='fbp'),
    url('skinview', views.skin_view, name='skinview'),
    url('mcloud/skin', views.rec_skin, name='recskin'),
    url('detectface', views.detect_face, name='detectface'),
    url('mcloud/statskin', views.stat_skin, name='statskin'),
]

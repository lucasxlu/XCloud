from django.conf.urls import url
from django.urls import path

from . import views

urlpatterns = [
    path('welcome', views.welcome, name='welcome'),
    path('index', views.index, name='index'),
    path('mcloud', views.mcloud, name='mcloud'),
    # path('mcloud/skin', views.rec_skin, name='recskin'),
    # path('sdr', views.sdr, name='sdr'),
    # url('facesearchview', views.face_search_view, name='facesearchview'),
    # url('facesearch', views.face_search, name='facesearch'),
    url('foodview', views.food_view, name='foodview'),
    url('food', views.food, name='food'),
    url('plantview', views.plant_view, name='plantview'),
    url('plant', views.plant, name='plant'),
    url('fbpview', views.fbp_view, name='fbpview'),
    url('fbp', views.fbp, name='fbp'),
    url('nsfwview', views.nsfw_view, name='nsfwview'),
    url('nsfw', views.nsfw, name='nsfw'),
    url('pdrview', views.pdr_view, name='pdrview'),
    url('pdr', views.pdr, name='pdr'),
    url('skinview', views.skin_view, name='skinview'),
    url('mcloud/skin', views.rec_skin, name='recskin'),
    # url('detectface', views.detect_face, name='detectface'),
    url('mcloud/statskin', views.stat_skin, name='statskin'),
    url('cbirview', views.cbir_view, name='cbirview'),
    url('cbir', views.cbir, name='cbir'),
    # url('deblurview', views.deblur_view, name='deblurview'),
    # url('deblur', views.deblur, name='deblur'),
]

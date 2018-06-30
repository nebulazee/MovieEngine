from django.conf.urls import url
from . import views

urlpatterns = [
    
    url(r'^$', views.mainPage, name='mainPage'),
    #url(r'^mainPage/$', views.mainPage, name='mainPage'),
    url(r'^goToRecom/controller$',views.movieRecom,name='movieRecom'),
    url(r'^goToRecom/$',views.index,name='index'),
]
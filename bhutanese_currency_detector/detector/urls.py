from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('detect/', views.detect_currency, name='detect_currency'),
    path('detect_camera/', views.detect_camera, name='detect_camera'),
]
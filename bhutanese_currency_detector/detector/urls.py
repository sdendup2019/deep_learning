# #/detector/urls.py

# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),
#     path('about/', views.about, name='about'),
#     path('detect/', views.detect_currency, name='detect_currency'),
#     path('detect_camera/', views.detect_camera, name='detect_camera'),
# ]
# /detector/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('detect/', views.detect_currency, name='detect_currency'),
    path('detect_camera/', views.detect_camera, name='detect_camera'),
    
    # Debug endpoints
    path('debug/model/', views.debug_model, name='debug_model'),
    path('debug/test/', views.test_model, name='test_model'),
]
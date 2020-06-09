from django.urls import path
from predictions import views

urlpatterns = [
    path('initialpreprocessing/', views.initialPreprocessing),
    path('emailStringPredection/', views.emailStringPredection),
    path('trainingModelForEvent/', views.trainingModelForEvent),
    path('googleConntect/', views.googleConntect),
    path('google/auth', views.google_authenticate),
]
from django.urls import path
from predictions import views
from rest_framework.authtoken.views import obtain_auth_token 
from .router import router
from django.urls import path, include

urlpatterns = [
    path('initialpreprocessing/', views.initialPreprocessing),
    path('emailStringPredection/', views.emailStringPredection),
    path('trainingModelForEvent/', views.trainingModelForEvent),
    path('google/authorizeUrl', views.authorizeUrl),
    path('google/auth', views.setCredentials),
    path('google/get_labels', views.getLabels),
    path('api-token-auth', obtain_auth_token, name='api_token_auth'),
    path('userData/', include(router.urls)),
]
from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
from django.db.models import Model 
# Created an empty model  
class Email(models.Model):
    sender = models.CharField(max_length=150)
    message_id = models.CharField(max_length=150)
    header = models.CharField(max_length=150)
    body = models.TextField()
    events = models.CharField(max_length=150)
    types = models.CharField(max_length=150)


class Users(AbstractUser):
    REQUIRED_FIELDS = ('user',)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    token = models.TextField()
    email = models.EmailField( unique=True)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []
    user_gmail_id = models.CharField(max_length=256)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)



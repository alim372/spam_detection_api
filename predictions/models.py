from django.db import models

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

from django.shortcuts import render
from emailStateDetection.apiResponse import prepareResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from predictions.initials import initialFunctions
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .models import Email
from .featureExtraction import EmailToWords
from .vectorization import WordCountToVector
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from .SVMtraining import SVMtraining
from .predictions import Prediction
from sklearn.feature_extraction.text import TfidfVectorizer
from .initials import initialFunctions


# 
@api_view(['POST'])
def initialPreprocessing(request):
    """
    initialPreprocessing function.
    """
    emailPipeline = Pipeline([
            ("Email to Words", EmailToWords()),
    ])
    # sender = request.POST['sender']
    receiver = request.POST['receiver']
    message_id = request.POST['message_id']
    subject = request.POST['subject']
    body = request.POST['body']
    event = request.POST['event']
    body = emailPipeline.fit_transform([body]) # stemming content into the original words and remove stop words
    subject = emailPipeline.fit_transform([subject]) # stemming content into the original words and remove stop words
    ins = initialFunctions()
    return Response(ins.create_df_traing(receiver, body, event))
    filename = str(message_id) + ".txt"
    content = {'id' : message_id, 'data' : { 'body' : [word for word in body.split(' ') if word !=""], 'subject': [word for word in subject.split(' ') if word !=""]},'event': event}
    return Response(prepareResponse([],[],content,  True ,'data processed successfully' , []) )

@api_view(['POST'])
def emailStringPredection(request):
    """
    email string Predection function.
    """
    sender   = request.POST['sender']
    receiver = request.POST['receiver']
    message_id = request.POST['message_id']
    header = request.POST['header']
    body = request.POST['body']
    fullContent =  header +' '+ body
    predObj = Prediction(body, receiver)
    result = predObj.predict()
    if (result[0] == 'spam'):
        emailInstance = Email.objects.create(
            sender= sender,
            message_id= message_id,
            header=header,
            types='spam',
        )
        return Response(prepareResponse([],[],result,  True ,'this email is spam' , []) )
    else:
        return Response(prepareResponse([],[],result,  True ,'this email is ham' , []) )

@api_view(['POST'])
def trainingModelForEvent(request):
    receiver = request.POST['receiver']
    event = request.POST['event']
    history = request.POST['history'] # json which contaion history messages
    path = 'data/models/' + receiver
    try:
        os.mkdir(path)
    except OSError:
        pass
    SVMobj = SVMtraining(history, path)
    SVMobj.training()
    return Response(prepareResponse([],[],[],  True ,'learn model... ' , []) )

    
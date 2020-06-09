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
#google connect
import pickle
import os.path
import googleapiclient.discovery
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from django.http import HttpResponseRedirect
import requests
import urllib
@api_view(['POST'])
def initialPreprocessing(request):
    """
    initialPreprocessing function.
    """
    emailPipeline = Pipeline([
        ("Email to Words", EmailToWords()),
    ])
    if (is_json(request.body)):
        postData = json.loads(request.body)
        # sender = request.POST['sender']
        receiver = postData['receiver']
        message_id = postData['message_id']
        subject = postData['subject']
        body = postData['body']
        event = postData['event']
    else:
        receiver = request.POST['receiver']
        message_id = request.POST['message_id']
        subject = request.POST['subject']
        body = request.POST['body']
        event = request.POST['event']

    
    # stemming content into the original words and remove stop words
    body = emailPipeline.fit_transform([body])
    # stemming content into the original words and remove stop words
    subject = emailPipeline.fit_transform([subject])
    ins = initialFunctions()
    ins.create_df_traing(receiver,subject, body, event,message_id )
    filename = str(message_id) + ".txt"
    content = {'id': message_id, 'data': {'body': [word for word in body.split(
        ' ') if word != ""], 'subject': [word for word in subject.split(' ') if word != ""]}, 'event': event}
    return Response(prepareResponse([], [], content,  True, 'data processed successfully', []))


@api_view(['POST'])
def emailStringPredection(request):
    """
    email string Predection function.
    """
    sender = request.POST['sender']
    receiver = request.POST['receiver']
    message_id = request.POST['message_id']
    header = request.POST['header']
    body = request.POST['body']
    fullContent = header + ' ' + body
    predObj = Prediction(body, receiver)
    result = predObj.predict()
    if (result[0] == 'spam'):
        # emailInstance = Email.objects.create(
        #     sender=sender,
        #     message_id=message_id,
        #     header=header,
        #     types='spam',
        # )
        return Response(prepareResponse([], [], result,  True, 'this email is spam', []))
    else:
        return Response(prepareResponse([], [], result,  True, 'this email is ham', []))


@api_view(['POST'])
def trainingModelForEvent(request):
    receiver = request.POST['receiver']
    event = request.POST['event']
    history = request.POST['history']  # json which contaion history messages
    path = 'data/models/' + receiver
    try:
        os.mkdir(path)
    except OSError:
        pass
    SVMobj = SVMtraining(history, path)
    SVMobj.training()
    return Response(prepareResponse([], [], [],  True, 'learn model... ', []))


@api_view(['GET'])
def googleConntect(request):
    token_request_uri = "https://accounts.google.com/o/oauth2/auth"
    response_type = "code"
    client_id = "699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com"
    redirect_uri = "https://run.ezinbox.app/google/auth"
    scope = "https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email"
    url = "{token_request_uri}?response_type={response_type}&client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}".format(
        token_request_uri = token_request_uri,
        response_type = response_type,
        client_id = client_id,
        redirect_uri = redirect_uri,
        scope = scope)
    return HttpResponseRedirect(url)

@api_view(['GET'])
def google_authenticate(request):
    
    login_failed_url = '/'
    if 'error' in request.GET or 'code' not in request.GET:
        return HttpResponseRedirect('{loginfailed}'.format(loginfailed = login_failed_url))

    access_token_uri = 'https://accounts.google.com/o/oauth2/token'
    redirect_uri = "https://run.ezinbox.app/predictions/google/authenticate/"
    params = dict(
        code=request.GET['code'],
        redirect_uri=redirect_uri,
        client_id="699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com",
        client_secret="tR0EzwK7VA1sZgJhjjNpHYnU",
        grant_type='authorization_code'
    )
    headers={'content-type':'application/x-www-form-urlencoded'}
    resp = requests.post(url=access_token_uri, params = params, headers = headers)
    token_data = resp.json() 
    return Response(token_data)
    resp = requests.get("https://www.googleapis.com/oauth2/v1/userinfo?access_token={accessToken}".format(accessToken=token_data['access_token']))
    #this gets the google profile!!
    google_profile = resp.json() 
    #log the user in-->
    #HERE YOU LOG THE USER IN, OR ANYTHING ELSE YOU WANT
    #THEN REDIRECT TO PROTECTED PAGE
    return HttpResponseRedirect('/dashboard')

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True



# If modifying these scopes, delete the file token.pickle.

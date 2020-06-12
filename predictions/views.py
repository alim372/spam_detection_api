from django.shortcuts import render
from emailStateDetection.apiResponse import prepareResponse
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
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
import pickle
import os.path
import googleapiclient.discovery
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from django.http import HttpResponseRedirect
import requests
import urllib
import httplib2
from googleapiclient.discovery import build
from oauth2client.file import Storage
from oauth2client import file, client, tools
from oauth2client import tools
from oauth2client.client import OAuth2WebServerFlow
# from predictions.auth import auth
import auth
from apiclient import errors
# from authentication.permissions import IsNotAuthenticated
from .models import Users

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

# If modifying these scopes, delete the file token.pickle.

@api_view(['get'])
# @permission_classes((IsNotAuthenticated,))
def authorizeUrl(request):
    setting = getFileJson('.credentials/application.json')
    flow = OAuth2WebServerFlow(
        client_id=setting['web']['client_id'],
        client_secret=setting['web']['client_secret'],
        scope=(getattr(settings, "SCOPS", None)),
        redirect_uri=getattr(
            settings, "HOST", None) + 'google/auth'
    )
    return Response(prepareResponse({}, {"url": flow.step1_get_authorize_url()}, True, 'data returned successfuly', []))


@api_view(['get'])
def setCredentials(request):
    setting = getFileJson('.credentials/application.json')
    code = request.GET['code']
    # create flow
    flow = OAuth2WebServerFlow(
        client_id=setting['web']['client_id'],
        client_secret=setting['web']['client_secret'],
        scope=(getattr(settings, "SCOPS", None)),
        redirect_uri=getattr(
            settings, "HOST", None) + 'google/auth'
    )
    # save credentials
    credentials = flow.step2_exchange(code)
    # get token
    token = credentials.access_token

    # get user info
    respUserInfo = requests.get(
        "https://www.googleapis.com/oauth2/v1/userinfo?access_token={accessToken}".format(accessToken=token))
    userInfoData = respUserInfo.json()

    # save user info
    setFile('.credentials/' + userInfoData['id'] +
            '.credentials.json', credentials)
    setFileJson('.credentials/' + userInfoData['id'] +
                '.userInfo.json', userInfoData)

    data = {
        'first_name' : userInfoData['first_name'],
        'last_name' : userInfoData['last_name'],
        'email' : userInfoData['email'],
    }
    ins = initialFunctions()
    ins.setUserData(data)
    return Response(prepareResponse({}, userInfoData, True, 'data returned successfuly', []))


@api_view(['get'])
def getLabels(request):
    user = Users.objects.get(token=request.POST['token'])
    if user:
        user_gmail_id = user.user_gmail_id
        credentials = getFile(
            '.credentials/'+user_gmail_id+'.credentials.json')
        service = build('gmail', 'v1', credentials=credentials)
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])

        return Response(prepareResponse({}, labels, True, 'data returned successfuly', []))
    else:
        return Response(prepareResponse({}, Newlabel, True, 'you are not autherize', []))



@api_view(['get'])
def setLabel(request):
    user = Users.objects.get(token=request.POST['token'])
    if user:
        user_gmail_id = user.user_gmail_id
        credentials = getFile(
            '.credentials/'+user_gmail_id+'.credentials.json')
        service = build('gmail', 'v1', credentials=credentials)
        Newlabel = MakeLabel('read later')
        CreateLabel(service, user_gmail_id, Newlabel)
        return Response(prepareResponse({}, Newlabel, True, 'data returned successfuly', []))
    else:
        return Response(prepareResponse({}, Newlabel, True, 'you are not autherize', []))



def CreateLabel(service, user_id, label_object):

    label = service.users().labels().create(
        userId=user_id, body=label_object).execute()
    return label


def MakeLabel(label_name, mlv='show', llv='labelShow'):
    label = {'messageListVisibility': mlv,
             'name': label_name,
             'labelListVisibility': llv}
    return label


def setFile(file, data):
    storage = Storage(file)
    storage.put(data)


def getFile(file):
    storage = Storage(file)
    return storage.get()


def setFileJson(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)


def getFileJson(file):
    with open(file) as f:
        data = json.load(f)
    return data


def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True




from .serializers import userSerializers
from .models import Users
from rest_framework import viewsets
class userviewsets(viewsets.ModelViewSet):
	queryset = Users.objects.all()
	serializer_class = userSerializers

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
    APPLICATION_NAME = 'ezinbox'
    SCOPES = 'https://mail.google.com/'
    CLIENT_SECRET_FILE = 'client_secret.json'
    authInst = auth.auth(SCOPES, CLIENT_SECRET_FILE, APPLICATION_NAME)
    creds = authInst.get_credentials() 
    store = file.Storage('token.json')
    if not creds or creds.invalid:
        flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
        creds = flow.authorization_url()
    service = build('gmail', 'v1', creds)
    return Response([])
    
    
    
    # token_request_uri = "https://accounts.google.com/o/oauth2/auth"
    # response_type = "code"
    # client_id = "699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com"
    # redirect_uri = "https://run.ezinbox.app/google/auth"
    # scope = "https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email https://mail.google.com/"
    # url = "{token_request_uri}?response_type={response_type}&client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}".format(
    #     token_request_uri = token_request_uri,
    #     response_type = response_type,
    #     client_id = client_id,
    #     redirect_uri = redirect_uri,
    #     scope = scope)

    # flow = InstalledAppFlow.from_client_secrets_file(
    #             'credentials.json', SCOPES)

    
    return HttpResponseRedirect(url)

@api_view(['GET'])
def google_authenticate(request):
    
    login_failed_url = '/'
    if 'error' in request.GET or 'code' not in request.GET:
        return HttpResponseRedirect('{loginfailed}'.format(loginfailed = login_failed_url))

    access_token_uri = 'https://oauth2.googleapis.com/token'
    redirect_uri = "https://run.ezinbox.app/google/auth"
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
    return Response( token_data)
    resp = requests.get("https://www.googleapis.com/oauth2/v1/userinfo?access_token={accessToken}".format(accessToken=token_data['access_token']))
    #this gets the google profile!!
    bodyParams ={
        "labelListVisibility": "labelShow",
        "messageListVisibility": "show",
        "name": "test"
        }
    google_profile = resp.json() 
    # return Response(google_profile)

    params2 = dict(
        code=request.GET['code'],
        redirect_uri=redirect_uri,
        client_id="699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com",
        client_secret="tR0EzwK7VA1sZgJhjjNpHYnU",
        grant_type='authorization_code', 
        access_token=token_data['access_token']
    )
    service = build_service(params2)
    return Response([True])
    createLabel = requests.post(url="https://www.googleapis.com/gmail/v1/users/107421768394579531195/labels", params = params2, data=bodyParams , headers = headers)
    return Response(createLabel.json())
    #log the user in-->
    #HERE YOU LOG THE USER IN, OR ANYTHING ELSE YOU WANT
    #THEN REDIRECT TO PROTECTED PAGE
    return Response(google_profile)
    return HttpResponseRedirect('/dashboard')

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True

def build_service(credentials):
  """Build a Gmail service object.

  Args:
    credentials: OAuth 2.0 credentials.

  Returns:
    Gmail service object.
  """
  http = httplib2.Http()
  http = credentials.authorize(http)
  return build('gmail', 'v1', http=http)


# If modifying these scopes, delete the file token.pickle.

@api_view(['get'])
def login(request):
    flow = OAuth2WebServerFlow(client_id="699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com",
                            client_secret="tR0EzwK7VA1sZgJhjjNpHYnU",
                            scope=('https://mail.google.com/ https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email '),
                            redirect_uri='https://run.ezinbox.app/google/auth')

    return HttpResponseRedirect(flow.step1_get_authorize_url())
    
# set credential
@api_view(['get'])
def auth(request):
    code = request.GET['code']
    flow = OAuth2WebServerFlow(client_id="699536180431-i1cqqn6nmoahdr135pnibsg8ghtca45q.apps.googleusercontent.com",
                            client_secret="tR0EzwK7VA1sZgJhjjNpHYnU",
                            scope=('https://mail.google.com/ https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email '),
                            redirect_uri='https://run.ezinbox.app/google/auth')
    credentials = flow.step2_exchange(code)
    token= credentials.access_token
    respUserInfo = requests.get("https://www.googleapis.com/oauth2/v1/userinfo?access_token={accessToken}".format(accessToken=token))
    
    setFile('credentials.json',credentials )
    setFileJson('userInfo.json',respUserInfo.json() )

    return Response(respUserInfo.json())

@api_view(['get'])
def get_labels(request):
    storage = Storage('credentials.json')
    credentials = storage.get()
    service = build('gmail', 'v1', credentials=credentials)
    results = service.users().labels().list(userId='me').execute()
    labels = results.get('labels', [])
    userData = getFileJson('userInfo.json')
    Newlabel =  MakeLabel('read later')
    label = CreateLabel(service, userData['id'], Newlabel)

    return Response(label)



def CreateLabel(service, user_id, label_object):
   
    label = service.users().labels().create(userId=user_id,body=label_object).execute()
    return label



def MakeLabel(label_name, mlv='show', llv='labelShow'):
    label = {'messageListVisibility': mlv,
            'name': label_name,
            'labelListVisibility': llv}
    return label

def setFile(file, data):
    storage = Storage(file)
    storage.put(data)

def setFileJson(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def getFileJson(file):
    with open(file) as f:
        data = json.load(f)
    return data
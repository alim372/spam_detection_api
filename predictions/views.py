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
# 
@api_view(['POST'])
def initialPreprocessing(request):
    """
    initialPreprocessing function.
    """

    sender = request.POST['sender']
    message_id = request.POST['message_id']
    header = request.POST['header']
    body = request.POST['body']
    events = request.POST['events']
    initFuns = initialFunctions()
    body = initFuns.stemming(body) # stemming content into the original words and remove stop words

    emailInstance = Email.objects.create(
        sender='sender',
        message_id='message_id',
        header='header',
        events='events',
        types='events',
        )
    
    filename = str(message_id) + ".txt"
    content = body
    response = HttpResponse(content, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename={0}'.format(filename)
    return response

    # return Response(prepareResponse([],[],[],  True ,body , []) )

@api_view(['POST'])
def emailFilePredection(request):
    """
    emailFilePredection function.
    """
    uploadedFile = request.FILES['file']
    fs = FileSystemStorage('data/temp/')
    
    filename = fs.save(uploadedFile.name , uploadedFile)

    initFuns = initialFunctions()
    testedEmail = initFuns.load_file_email(uploadedFile.name )
    emailPipeline = Pipeline([
        ("Email to Words", EmailToWords()),
        ("Wordcount to Vector", WordCountToVector()),
    ])

    XAugmentedTest = emailPipeline.transform([testedEmail])
    modelName = 'data/models/finalized_model.sav'
    loadedModel = joblib.load(modelName)
    result = loadedModel.predict(XAugmentedTest)
    
    return Response(prepareResponse(result,[],[],  True ,'email_pipeline' , []) )

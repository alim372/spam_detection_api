import pandas as pd
from nltk import stem
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import json
from sklearn.pipeline import Pipeline
from .featureExtraction import EmailToWords
import joblib

class SVMtraining:
    dfLearning = pd.DataFrame(columns=['text', 'label'])
    path = ''
    def __init__(self, data, path):
        self.path = path 
        jsonHistory = json.loads(data)
        messages = []
        labels = []
        for entry in jsonHistory:
            message = ''
            for word in entry['data'] : 
                message += word+' '
            labels.append(entry['label'])
            messages.append(message)
        self.dfLearning['text'] = messages
        self.dfLearning['label'] = labels
    
    def training(self):
        vectorizer = TfidfVectorizer()
        X_train = self.dfLearning['text']
        y_train = self.dfLearning['label']
        X_train = vectorizer.fit_transform(X_train)
        from sklearn import svm
        svm = svm.SVC(C=1000)
        svm.fit(X_train, y_train)
        joblib.dump(svm, self.path+'/predictionSVM.sav')
        joblib.dump(vectorizer, self.path+'/vectorizerSVM.sav')
    
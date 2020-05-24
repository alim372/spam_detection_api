import pandas as pd
import numpy as np
import os
import email
import email.policy
from bs4 import BeautifulSoup
from collections import Counter
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
import joblib
from .initials import initialFunctions

class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self, stripHeaders=True, lowercaseConversion = True, punctuationRemoval = True, 
                 urlReplacement = True, numberReplacement = True, stemming = True):
        self.stripHeaders = stripHeaders
        self.lowercaseConversion = lowercaseConversion
        self.punctuationRemoval = punctuationRemoval
        self.urlReplacement = urlReplacement
        #self.url_extractor = urlextract.URLExtract()
        self.numberReplacement = numberReplacement
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_to_words = []
        for email in X:
            if (type(email) != str):
                ins = initialFunctions()
                text = ins.email_to_plain( email)
            else:
                text = email
            if text is None:
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()
                
            #if self.urlReplacement:
                #urls = self.url_extractor.find_urls(text)
                #for url in urls:
                #    text = text.replace(url, 'URL')   
                    
            if self.punctuationRemoval:
                text = text.replace('.','')
                text = text.replace(',','')
                text = text.replace('!','')
                text = text.replace('?','')
                
            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)
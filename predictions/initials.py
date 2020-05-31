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
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

class initialFunctions:
    def get_email_structure(self, email):
        if isinstance(email, str):
            return email
        payload = email.get_payload()
        if isinstance(payload, list):
            return "multipart({})".format(", ".join([
                self.get_email_structure(sub_email)
                for sub_email in payload
            ]))
        else:
            return email.get_content_type()

    def structures_counter(self,emails):
        structures = Counter()
        for email in emails:
            structure = self.get_email_structure(email)
            structures[structure] += 1
        return structures

    def html_to_plain(self,contnet):
        try:
            soup = BeautifulSoup(contnet, 'html.parser')
            return soup.text.replace('\n\n','')
        except:
            return "empty"

    def email_to_plain(self,email):
        struct = self.get_email_structure(email)
        for part in email.walk():
            partContentType = part.get_content_type()
            if partContentType not in ['text/plain','text/html']:
                continue
            try:
                partContent = part.get_content()
            except: # in case of encoding issues
                partContent = str(part.get_payload())
            if partContentType == 'text/plain':
                return partContent
            else:
                return self.html_to_plain(part)

    def stemming(self, emailBody):
        body = self.html_to_plain(emailBody)
        stop_words = set(stopwords.words('english')) 
        
        word_tokens = word_tokenize(body) 
        
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        
        filtered_sentence = [] 
        
        stemmer = nltk.PorterStemmer()

        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(stemmer.stem(w)) 
        return ' '.join(filtered_sentence)

    def load_file_email(self, filename):
        directory = "data/temp/" 
        with open(os.path.join(directory, filename), "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)   

    def load_string_email(self, stringEmail):
        return email.message_from_string(stringEmail, policy=email.policy.default)

    def load_email(is_spam, filename):
        directory = "data/temp/spam" if is_spam else "data/temp/ham"
        with open(os.path.join(directory, filename), "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)

from sklearn.pipeline import Pipeline
import joblib
from .featureExtraction import EmailToWords

class Prediction:
    content = ''
    path = 'data/models/'
    def __init__(self, body, receiver):
        emailPipeline = Pipeline([
            ("Email to Words", EmailToWords()),
        ])
        self.content = emailPipeline.fit_transform([body])
        self.path = self.path + '/' +receiver

    def predict(self):
        loaded_model = joblib.load(self.path + '/' + 'predictionSVM.sav')
        loaded_vectorizer = joblib.load(self.path + '/' +'vectorizerSVM.sav')
        vectorizedContent = loaded_vectorizer.transform([self.content])
        return loaded_model.predict(vectorizedContent)
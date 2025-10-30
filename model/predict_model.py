import joblib
import numpy as np

class SpamClassifier:
    def __init__(self):
        self.model = joblib.load("model/spam_model.pkl")
        self.embedder = joblib.load("model/bert_embedder.pkl")

    def predict(self, email_text):
        embedding = self.embedder.encode([email_text])
        prediction = self.model.predict(embedding)[0]
        label = "Spam" if prediction == 1 else "Ham"
        return label

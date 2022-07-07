import pickle
import os

from sklearn.preprocessing import StandardScaler

from hardware_language_library_extractor.extras.config import *


class Classifier:
    def __init__(self):
        self.hardware_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, HARDWARE_CLASSIFIER_MODEL), 'rb'))
        self.language_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LANGUAGE_CLASSIFIER_MODEL), 'rb'))
        self.library_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LIBRARY_CLASSIFIER_MODEL), 'rb'))
        self.scaler = StandardScaler()

    def get_hardware_predictions(self, sentence_embeddings):
        sentence_embeddings = self.scaler.fit_transform(sentence_embeddings)
        return self.hardware_model.predict(sentence_embeddings)

    def get_language_predictions(self, sentence_embeddings):
        sentence_embeddings = self.scaler.fit_transform(sentence_embeddings)
        return self.language_model.predict(sentence_embeddings)

    def get_library_predictions(self, sentence_embeddings):
        sentence_embeddings = self.scaler.fit_transform(sentence_embeddings)
        return self.library_model.predict(sentence_embeddings)

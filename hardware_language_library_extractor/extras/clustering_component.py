from hardware_language_library_extractor.extras.config import *
import pickle
import os


class Clustering:
    def __init__(self):
        self.hardware_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, HARDWARE_CLUSTERING_MODEL), 'rb'))
        self.language_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LANGUAGE_CLUSTERING_MODEL), 'rb'))
        self.library_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LIBRARY_CLUSTERING_MODEL), 'rb'))

    def get_hardware_clusters(self, sentence_embeddings):
        return self.hardware_model.predict(sentence_embeddings)

    def get_language_clusters(self, sentence_embeddings):
        return self.language_model.predict(sentence_embeddings)

    def get_library_clusters(self, sentence_embeddings):
        return self.library_model.predict(sentence_embeddings)

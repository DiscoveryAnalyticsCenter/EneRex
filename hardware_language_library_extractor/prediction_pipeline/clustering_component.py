from hardware_language_library_extractor.prediction_pipeline.config import MODELS_FOLDER_BASE_PATH, \
    HARDWARE_CLUSTERING_MODEL, LANGUAGE_CLUSTERING_MODEL, LIBRARY_CLUSTERING_MODEL
from hardware_language_library_extractor.common.embedding_component import Embeddings
from typing import Dict

import pickle
import os


class Clustering:
    def __init__(self):
        self.hardware_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, HARDWARE_CLUSTERING_MODEL), 'rb'))
        self.lang_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LANGUAGE_CLUSTERING_MODEL), 'rb'))
        self.lib_model = pickle.load(open(os.path.join(MODELS_FOLDER_BASE_PATH, LIBRARY_CLUSTERING_MODEL), 'rb'))
        self.cluster_specific_embedding_component = Embeddings()

    def get_cluster_predictions(self, doc: Dict):
        doc["hardware_cluster"] = False
        doc["lang_cluster"] = False
        doc["lib_cluster"] = False
        if doc["has_hardware"] or doc["has_lang"] or doc["has_lib"]:
            scierc_embeddings = self.cluster_specific_embedding_component.get_embeddings_scierc([doc["sentence"]])
            if doc["has_hardware"]:
                tagged_h_cluster = self.hardware_model.predict(scierc_embeddings)[0]
                if tagged_h_cluster == 0:
                    doc["hardware_cluster"] = True
            if doc["has_lang"]:
                tagged_lang_cluster = self.lang_model.predict(scierc_embeddings)[0]
                if tagged_lang_cluster == 0 or tagged_lang_cluster == 1:
                    doc["lang_cluster"] = True
            if doc["has_lib"]:
                tagged_lib_cluster = self.lib_model.predict(scierc_embeddings)[0]
                if tagged_lib_cluster == 0 or tagged_lib_cluster == 1:
                    doc["lib_cluster"] = True
        return doc

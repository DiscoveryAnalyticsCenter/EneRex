import os
from typing import Dict

from allennlp.data.dataset_readers import DatasetReader
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.common.from_params import Params
import torch
from spacy.gold import offsets_from_biluo_tags

from hardware_language_library_extractor.prediction_pipeline.config import MODELS_FOLDER_BASE_PATH, NER_MODEL, \
    MAX_THRESHOLD_SENTCHAR_LEN
from hardware_language_library_extractor.common.bert_crf_tagger import *
from hardware_language_library_extractor import n_gpu


class NER:
    def __init__(self):
        model_config = Params.from_file(os.path.join(MODELS_FOLDER_BASE_PATH, NER_MODEL, 'config.json'))
        self.model = Model.load(config=model_config, serialization_dir=os.path.join(MODELS_FOLDER_BASE_PATH, NER_MODEL))
        self.dataset_reader = DatasetReader.from_params(model_config.pop("dataset_reader"))
        self.predictor = SentenceTaggerPredictor(model=self.model, dataset_reader=self.dataset_reader)

    def get_entities(self, doc: Dict):
        entities = []
        if doc["hardware_cluster"] or doc["lang_cluster"] or doc["lib_cluster"]:
            if len(doc["sentence"]) > MAX_THRESHOLD_SENTCHAR_LEN:
                doc["sentence"] = doc["sentence"][:MAX_THRESHOLD_SENTCHAR_LEN]
            prediction = self.predictor.predict(doc["sentence"])
            spacy_doc = self.predictor._tokenizer.spacy(doc["sentence"])
            entities = offsets_from_biluo_tags(spacy_doc, prediction["tags"])
            for i in range(len(entities)):
                entities[i] = entities[i] + (spacy_doc.char_span(entities[i][0], entities[i][1]).text,)
        return entities

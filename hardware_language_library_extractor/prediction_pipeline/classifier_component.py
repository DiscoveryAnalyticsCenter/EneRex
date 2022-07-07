from transformers import AutoTokenizer, AutoModelForSequenceClassification, Pipeline
import torch
import os
from hardware_language_library_extractor.prediction_pipeline.config import MODELS_FOLDER_BASE_PATH, \
    TRANSFORMER_HARDWARE_CLASSIFIER, TRANSFORMER_LANGUAGE_CLASSIFIER, TRANSFORMER_LIBRARY_CLASSIFIER, \
    SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD, SCIBERT_EMBEDDING_MODEL
from hardware_language_library_extractor import device, n_gpu


class TransformerProcessing:
    def __init__(self):
        self.h_sent_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODELS_FOLDER_BASE_PATH, TRANSFORMER_HARDWARE_CLASSIFIER))
        self.lang_sent_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODELS_FOLDER_BASE_PATH, TRANSFORMER_LANGUAGE_CLASSIFIER))
        self.lib_sent_model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(MODELS_FOLDER_BASE_PATH, TRANSFORMER_LIBRARY_CLASSIFIER))
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_FOLDER_BASE_PATH, SCIBERT_EMBEDDING_MODEL))
        # self.h_nlp = Pipeline(task="sentiment-analysis", model=self.h_sent_model, tokenizer=self.tokenizer,
        # framework='pt', device=device_id) self.lang_nlp = Pipeline(task="sentiment-analysis",
        # model=self.lang_sent_model, tokenizer=self.tokenizer, framework='pt', device=device_id) self.lib_nlp =
        # Pipeline(task="sentiment-analysis", model=self.lib_sent_model, tokenizer=self.tokenizer, framework='pt',
        # device=device_id)
        if n_gpu > 1:
            self.h_sent_model = self.h_sent_model.to(device)
            self.h_sent_model = torch.nn.DataParallel(self.h_sent_model)
            self.lang_sent_model = self.lang_sent_model.to(device)
            self.lang_sent_model = torch.nn.DataParallel(self.lang_sent_model)
            self.lib_sent_model = self.lib_sent_model.to(device)
            self.lib_sent_model = torch.nn.DataParallel(self.lib_sent_model)

    def get_sentence_predictions(self, pdf_in_json_tokenized):
        tokenized_pdf_with_sentence_cats = dict()
        for key, value in pdf_in_json_tokenized.items():
            tokenized_pdf_with_sentence_cats[key] = self._get_preditctions([val.text for val in value]) if value else []
        return tokenized_pdf_with_sentence_cats

    def _get_preditctions(self, sentences):
        encodings = self.tokenizer(sentences, truncation=True, max_length=150, padding=True, return_tensors='pt').to(
            device)
        h_outputs = self.h_sent_model(**encodings)
        lang_outputs = self.lang_sent_model(**encodings)
        lib_outputs = self.lib_sent_model(**encodings)
        h_predictions = torch.softmax(h_outputs[0], dim=1).tolist()
        lang_predictions = torch.softmax(lang_outputs[0], dim=1)
        lib_predictions = torch.softmax(lib_outputs[0], dim=1)
        result = []
        for i in range(len(sentences)):
            result.append({"sentence": sentences[i], "has_hardware": False, "has_lang": False, "has_lib": False})
            if h_predictions[i][1] > SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD:
                result[i]["has_hardware"] = True
            if lang_predictions[i][1] > SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD:
                result[i]["has_lang"] = True
            if lib_predictions[i][1] > SENTENCE_CLASSIFIER_PROBABILITY_THRESHOLD:
                result[i]["has_lib"] = True
        return result

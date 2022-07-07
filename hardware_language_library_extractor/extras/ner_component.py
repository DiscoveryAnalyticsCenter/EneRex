import spacy
from spacy.pipeline import Sentencizer, EntityRuler
import torch


class NER:
    def __init__(self, hardware_keywords):
        spacy.util.fix_random_seed(0)
        is_using_gpu = spacy.prefer_gpu()
        if is_using_gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.remove_pipe('tagger')
        self.nlp.remove_pipe('parser')
        self.nlp.add_pipe(Sentencizer())
        self.ruler = EntityRuler(self.nlp, overwrite_ents=True)
        self.ruler.add_patterns(self.get_patterns(hardware_keywords))
        self.nlp.add_pipe(self.ruler)

    def get_ners(self, sentence):
        ner_sentence = self.nlp(sentence)
        return ner_sentence

    def get_patterns(self, hardware_keywords):
        patterns = []
        for item in hardware_keywords:
            patterns.append({"label": "hardware", "pattern": item})
        return patterns

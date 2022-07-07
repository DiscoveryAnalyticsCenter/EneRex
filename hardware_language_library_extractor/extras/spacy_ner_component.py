import spacy
from spacy.pipeline import Sentencizer


class NER:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.remove_pipe('tagger')
        self.nlp.remove_pipe('parser')
        self.nlp.add_pipe(Sentencizer())

    def get_ners(self, sentences):
        ner_sentences = [sent for sent in self.nlp.pipe(sentences)]
        return ner_sentences

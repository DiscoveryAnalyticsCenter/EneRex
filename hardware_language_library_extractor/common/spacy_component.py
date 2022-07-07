import spacy
from spacy.pipeline import Sentencizer


class SpacyProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(Sentencizer(), first=True)

    def get_tokenized_ssentences(self, pdf_in_json):
        pdf_in_json_tokenized = dict()
        with self.nlp.disable_pipes('ner', 'tagger', 'parser'):
            for key, value in pdf_in_json[1].items():
                pdf_in_json_tokenized[key] = list(self.nlp(value).sents)
        return pdf_in_json_tokenized

    def get_ssentences(self, paragraph):
        docs = self.nlp(paragraph)
        return [sent for sent in docs.sents]

    def get_words_from_spacy_sentences(self, sentences):
        with self.nlp.disable_pipes('ner', 'tagger', 'parser'):
            sents = []
            for sent in sentences:
                sents.append([token.text for token in sent])
            return sents

    def get_words_from_text_sentences(self, sentences):
        with self.nlp.disable_pipes('ner', 'tagger', 'parser'):
            sentences = self.nlp.pipe(sentences)
        return self.get_words_from_spacy_sentences(sentences)

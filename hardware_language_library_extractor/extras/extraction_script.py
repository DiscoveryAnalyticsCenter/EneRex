from hardware_language_library_extractor.common.util import *
from hardware_language_library_extractor.extras.config import *
from hardware_language_library_extractor.common.embedding_component import Embeddings
from hardware_language_library_extractor.extras.lr_based_classifier_component import Classifier
from hardware_language_library_extractor.extras.clustering_component import Clustering
from hardware_language_library_extractor.common.spacy_component import SpacyProcessor
from hardware_language_library_extractor.extras.spacy_ner_component import NER
from hardware_language_library_extractor.logger import Logger

logger = Logger(LOGGER_NAME, LOG_LEVEL).logger


class ProcessPDFs:
    def __init__(self):
        # self.pdf_paths = readCSV(os.path.join(PDF_FOLDER_PATH, LIST_OF_PDFs), header = [0])[0]
        self.sentence_tokenizer = SpacyProcessor()
        self.embedding_component = Embeddings()
        self.sentence_classifier = Classifier()
        self.cluster_component = Clustering()
        self.ner_component = NER()

    def get_positive_sentences_with_ners(self, sentence_embeddings, sentences, classifier, clusterer, section):
        classifier_sent_indexes = classifier(sentence_embeddings)
        classifier_positive_sent_indexes = [i for i in range(len(classifier_sent_indexes)) if
                                            classifier_sent_indexes[i] == 1]
        if not classifier_positive_sent_indexes:
            return [], [], []
        sentence_clusters = clusterer(
            [sentence_embeddings[i] for i in range(len(sentence_embeddings))
             if i in classifier_positive_sent_indexes])
        cluster_positive_sent_indexes = [classifier_positive_sent_indexes[i] for i in range(len(sentence_clusters)) if
                                         sentence_clusters[i] == 0]
        if not classifier_positive_sent_indexes:
            return [], [], []
        filtered_sentences = [sentences[i].text for i in cluster_positive_sent_indexes]
        ner_sentences = self.ner_component.get_ners(filtered_sentences)
        sent_units = []
        for i in range(len(ner_sentences)):
            sent_start = sentences[cluster_positive_sent_indexes[i] - 1].end_char if cluster_positive_sent_indexes[
                                                                                         i] - 1 >= 0 else 0
            sent_end = sentences[classifier_positive_sent_indexes[i] + 1].start_char if cluster_positive_sent_indexes[
                                                                                            i] + 1 < len(sentences) \
                else sentences[cluster_positive_sent_indexes[i]].end_char
            sent_units.extend(self.get_sentence_unit(ner_sentences[i], section, sent_start, sent_end))
        return sent_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes
        # word_sentences = self.sentence_tokenizer.get_words_from_spacy_sentences(
        #     [sentences[i] for i in positive_sent_indexes])
        # return word_sentences

    def get_sentence_unit(self, sentence, section, sent_start, sent_end):
        sent_units = []
        for ent in sentence.ents:
            sent_units.append(
                get_sentence_unit_skeleton(sentence.text, section, sent_start, sent_end, ent.start_char, ent.end_char,
                                           ent.text))
        return sent_units

    def get_hardware_ner_sentences(self, sentence_embeddings, sentences, section):
        ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes = \
            self.get_positive_sentences_with_ners(
                sentence_embeddings, sentences, self.sentence_classifier.get_hardware_predictions,
                self.cluster_component.get_hardware_clusters, section
            )
        return ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes

    def get_language_ner_sentences(self, sentence_embeddings, sentences, section):
        ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes = \
            self.get_positive_sentences_with_ners(
                sentence_embeddings, sentences,
                self.sentence_classifier.get_language_predictions,
                self.cluster_component.get_language_clusters, section
            )
        return ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes

    def get_library_ner_sentences(self, sentence_embeddings, sentences, section):
        ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes = \
            self.get_positive_sentences_with_ners(
                sentence_embeddings, sentences, self.sentence_classifier.get_library_predictions,
                self.cluster_component.get_library_clusters, section
            )
        return ner_sentence_units, classifier_positive_sent_indexes, cluster_positive_sent_indexes

    def get_intermediate_output(self, h_indexes, lg_indexes, lb_indexes, ssentences):
        section_result = []
        section_result.append(get_intermediate_output_unit(h_indexes, ssentences, "hardware"))
        section_result.append(get_intermediate_output_unit(lg_indexes, ssentences, "language"))
        section_result.append(get_intermediate_output_unit(lb_indexes, ssentences, "library"))
        return section_result

    def process_single_pdf(self, pdf_path):
        data = loadDataFromJSONFile(pdf_path)
        output = get_output_skeleton(data)
        classifier_output = dict()
        cluster_output = dict()
        for key, value in data[1].items():
            sentences = self.sentence_tokenizer.get_ssentences(value)
            sentence_embeddings = self.embedding_component.get_embeddings_scierc([sent.text for sent in sentences])
            hardware_output, h_classifier_result, h_cluster_result = self.get_hardware_ner_sentences(
                sentence_embeddings, sentences, key)
            if hardware_output:
                output['hardware'].extend(hardware_output)
            language_output, lg_classifier_result, lg_cluster_result = self.get_language_ner_sentences(
                sentence_embeddings, sentences, key)
            if language_output:
                output['language'].extend(language_output)
            library_output, lb_classifier_result, lb_cluster_result = self.get_library_ner_sentences(
                sentence_embeddings, sentences, key)
            if library_output:
                output['library'].extend(library_output)
            if h_classifier_result or lg_classifier_result or lb_classifier_result:
                classifier_output[key] = self.get_intermediate_output(h_classifier_result, lg_classifier_result,
                                                                      lb_classifier_result, sentences)
            if h_cluster_result or lg_cluster_result or lb_cluster_result:
                cluster_output[key] = self.get_intermediate_output(h_cluster_result, lg_cluster_result,
                                                                   lb_cluster_result, sentences)
        return output, classifier_output, cluster_output


def main():
    pdf_processor = ProcessPDFs()
    file_name = '1609.02226.json'
    pdf_path = os.path.join(PDF_FOLDER_PATH, file_name)
    output, classifier_output, cluster_output = pdf_processor.process_single_pdf(pdf_path)
    writeOutputToJSONFile(output, os.path.join(PDF_FOLDER_PATH, '{}.output.json'.format(file_name.split('.')[0])))
    writeOutputToJSONFile(output, os.path.join(PDF_FOLDER_PATH, '{}.classifier.json'.format(file_name.split('.')[0])))
    writeOutputToJSONFile(output, os.path.join(PDF_FOLDER_PATH, '{}.cluster.json'.format(file_name.split('.')[0])))
    print(output)


if __name__ == '__main__':
    main()

import os
from multiprocessing import Pool, set_start_method

from hardware_language_library_extractor.data_generator.config import HARDWARE_KEYWORDS, LANGUAGE_KEYWORDS, \
    LIBRARY_KEYWORDS, PDF_FOLDER_PATH, TRAINING_DATA_BASE_PATH, HARDWARE_NEGATIVE_SENTENCES, \
    LIBRARY_NEGATIVE_SENTENCES, LANGUAGE_NEGATIVE_SENTENCES, LIST_OF_PDFs
from hardware_language_library_extractor.common.util import get_regex_pattern, load_data_from_json, \
    append_multiple_lines_to_text_file, read_txt_into_df, shuffle_list
from hardware_language_library_extractor.common.spacy_component import SpacyProcessor
from hardware_language_library_extractor.data_generator import logger


class SentenceExtractor:
    def __init__(self):
        self.sentence_tokenizer = SpacyProcessor()
        self.hardware_regex = get_regex_pattern(HARDWARE_KEYWORDS)
        self.language_regex = get_regex_pattern(LANGUAGE_KEYWORDS)
        self.library_regex = get_regex_pattern(LIBRARY_KEYWORDS)

    def generate_negative_sentence(self, pdf_path):
        try:
            logger.debug("Processing started for the file: {}".format(pdf_path))
            pdf_in_json = load_data_from_json(os.path.join(PDF_FOLDER_PATH, pdf_path))
            tokenized_data = self.sentence_tokenizer.get_tokenized_ssentences(pdf_in_json)
            negative_hardware_sentences = list()
            negative_language_sentences = list()
            negative_library_sentences = list()
            for key, value in tokenized_data.items():
                for sent in value:
                    if not self.hardware_regex.findall(sent.text):
                        negative_hardware_sentences.append(sent.text)
                    if not self.language_regex.findall(sent.text):
                        negative_language_sentences.append(sent.text)
                    if not self.library_regex.findall(sent.text):
                        negative_library_sentences.append(sent.text)

            append_multiple_lines_to_text_file(negative_hardware_sentences,
                                               os.path.join(TRAINING_DATA_BASE_PATH, HARDWARE_NEGATIVE_SENTENCES))
            append_multiple_lines_to_text_file(negative_library_sentences,
                                               os.path.join(TRAINING_DATA_BASE_PATH, LIBRARY_NEGATIVE_SENTENCES))
            append_multiple_lines_to_text_file(negative_language_sentences,
                                               os.path.join(TRAINING_DATA_BASE_PATH, LANGUAGE_NEGATIVE_SENTENCES))
            logger.debug("Processing completed for the file: {}".format(pdf_path))
        except Exception as e:
            logger.error("Processing failed for the file: {} and the error is {}".format(pdf_path, e.args))


def init():
    global extractor
    extractor = SentenceExtractor()


def driver(pdf):
    extractor.generate_negative_sentence(pdf)


def main():
    try:
        pdfs = read_txt_into_df(LIST_OF_PDFs)[0]
        pdfs = shuffle_list(pdfs)[:2050]
        set_start_method('spawn')
        with Pool(5, initializer=init) as pool:
            pool.map(driver, pdfs)
    except RuntimeError as e:
        logger.error(e.args)


if __name__ == '__main__':
    main()

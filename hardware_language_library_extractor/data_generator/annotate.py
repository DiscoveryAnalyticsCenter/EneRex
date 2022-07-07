from multiprocessing import set_start_method, Pool
import os

from hardware_language_library_extractor.data_generator.config import PDF_FOLDER_PATH, TRAINING_DATA_BASE_PATH, \
    ANNOTATED_FOLDER, LIST_OF_PDFs
from hardware_language_library_extractor.common.util import load_data_from_json, get_output_skeleton, \
    write_output_to_json, get_hardware_annotation_regex, read_txt_into_df, create_folder
from hardware_language_library_extractor.common.spacy_component import SpacyProcessor
from hardware_language_library_extractor.logger import Logger

logger = Logger("annotation").logger


def process_single_pdf(file_name):
    data = load_data_from_json(os.path.join(PDF_FOLDER_PATH, file_name))
    output = get_output_skeleton(data)
    tokenized_data = spacy_processor.get_tokenized_ssentences(data)
    for key, value in tokenized_data.items():
        for sent in value:
            entities = pattern.findall(sent.text)
            if entities:
                output['hardware'].append({"sent": sent.text, "entities": list(set(entities))})
    if output['hardware']:
        write_output_to_json(output, os.path.join(TRAINING_DATA_BASE_PATH, ANNOTATED_FOLDER, file_name))


def init():
    global spacy_processor
    global pattern
    pattern = get_hardware_annotation_regex()
    spacy_processor = SpacyProcessor()


def main():
    file_list = read_txt_into_df(LIST_OF_PDFs)[0]
    create_folder(TRAINING_DATA_BASE_PATH, ANNOTATED_FOLDER)
    try:
        set_start_method('spawn')
        with Pool(processes=3, initializer=init) as pool:
            pool.map(process_single_pdf, file_list)
    except RuntimeError as e:
        logger.error(e.args)
        pass


if __name__ == '__main__':
    main()

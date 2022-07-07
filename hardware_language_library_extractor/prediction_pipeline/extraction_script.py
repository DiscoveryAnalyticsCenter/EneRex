import os

from hardware_language_library_extractor.prediction_pipeline.config import PDF_FOLDER_PATH, OUTPUT_FOLDER, LIST_OF_PDFs
from hardware_language_library_extractor.common.util import get_sentence_unit_skeleton, \
    load_data_from_json, write_output_to_json, get_output_skeleton, read_txt_into_df, create_folder
from hardware_language_library_extractor.prediction_pipeline.clustering_component import Clustering
from hardware_language_library_extractor.prediction_pipeline.bert_crf_ner_component import NER
from hardware_language_library_extractor.common.spacy_component import SpacyProcessor
from hardware_language_library_extractor.prediction_pipeline.classifier_component import TransformerProcessing
from hardware_language_library_extractor.prediction_pipeline import logger


class ProcessPDFs:
    def __init__(self):
        self.transformer_processor = TransformerProcessing()
        self.clusterer = Clustering()
        self.ner_tagger = NER()
        self.spacy_processor = SpacyProcessor()

    def get_outputs(self, tokenized_pdf_with_sentence_cats, tokenized_data, output_skeleton):
        intermediate_output = dict()
        for key, value in tokenized_pdf_with_sentence_cats.items():
            intermediate_output[key] = []
            hardware_platforms_output_units = []
            language_libraries_output_units = []
            compute_resources_output_units = []
            for i in range(len(tokenized_data[key])):
                doc = value[i]
                doc = self.clusterer.get_cluster_predictions(doc)
                if doc["has_hardware"] or doc["has_lang"] or doc["has_lib"]:
                    intermediate_output[key].append(doc)
                hardware_platforms = []
                language_libraries = []
                compute_resources = []
                for entity in self.ner_tagger.get_entities(doc):
                    if doc["has_hardware"] and entity[2] == 'hardware_platform':
                        hardware_platforms.append(entity)
                    if doc["has_hardware"] and (entity[2] == "hardware_resources" or entity[2] == "compute_time"):
                        compute_resources.append(entity)
                    if doc["has_lang"] and entity[2] == 'p_language':
                        language_libraries.append(entity)
                    if doc["has_lib"] and entity[2] == 'p_library':
                        language_libraries.append(entity)
                if hardware_platforms:
                    hardware_platforms_output_units.append(get_sentence_unit_skeleton(doc["sentence"], key,
                                                                                      tokenized_data[key][i].start_char,
                                                                                      tokenized_data[key][i].end_char,
                                                                                      hardware_platforms))
                if language_libraries:
                    language_libraries_output_units.append(get_sentence_unit_skeleton(doc["sentence"], key,
                                                                                      tokenized_data[key][i].start_char,
                                                                                      tokenized_data[key][i].end_char,
                                                                                      language_libraries))
                if compute_resources:
                    compute_resources_output_units.append(get_sentence_unit_skeleton(doc["sentence"], key,
                                                                                     tokenized_data[key][i].start_char,
                                                                                     tokenized_data[key][i].end_char,
                                                                                     compute_resources))
            if hardware_platforms_output_units:
                output_skeleton["hardware_platforms"].extend(hardware_platforms_output_units)
            if language_libraries_output_units:
                output_skeleton["language_libraries"].extend(language_libraries_output_units)
            if compute_resources_output_units:
                output_skeleton["compute_resources"].extend(compute_resources_output_units)

        return output_skeleton, intermediate_output

    def process_single_pdf(self, pdf_path):
        data = load_data_from_json(pdf_path)
        output = get_output_skeleton(data)
        tokenized_data = self.spacy_processor.get_tokenized_ssentences(data)
        tokenized_pdf_with_sentence_cats = self.transformer_processor.get_sentence_predictions(tokenized_data)
        output, intermediate_output = self.get_outputs(tokenized_pdf_with_sentence_cats, tokenized_data, output)
        return output, intermediate_output


def driver(file_name, pdf_processor):
    assert pdf_processor is not None
    logger.info("Processing started for file {}".format(file_name))
    try:
        pdf_path = os.path.join(PDF_FOLDER_PATH, file_name)
        output, intermediate_output = pdf_processor.process_single_pdf(pdf_path)
        write_output_to_json(output, os.path.join(OUTPUT_FOLDER, '{}.output.json'.format(file_name.split('.json')[0])))
        write_output_to_json(intermediate_output, os.path.join(OUTPUT_FOLDER, '{}.intermediate_output.json'.format(
            file_name.split('.json')[0])))
        logger.info("Processing completed for file {}".format(file_name))
    except Exception as e:
        logger.error("Processing failed for file {} and error arguments are {}".format(file_name, e.args))


def main():
    file_list = read_txt_into_df(LIST_OF_PDFs)[0]
    file_list = file_list[:1000] if len(file_list) > 1000 else file_list
    create_folder(OUTPUT_FOLDER, '', True)
    pdf_processor = ProcessPDFs()
    for file in file_list:
        driver(file, pdf_processor)


if __name__ == '__main__':
    main()

import os

from hardware_language_library_extractor.common.spacy_component import SpacyProcessor
from hardware_language_library_extractor.data_generator.config import TRAINING_DATA_BASE_PATH, \
    LANGUAGE_CLUSTER_MAPPING_FILE, LIBRARY_CLUSTER_MAPPING_FILE
from hardware_language_library_extractor.common.util import read_txt_into_df


def get_data(from_csv, hardware_cluster_to_be_used, language_cluster_to_be_used, library_cluster_to_be_used):
    hardware_data, lang_data, lib_data = [], [], []
    if from_csv:
        # hardware_data = readCSV(os.path.join(TRAINING_DATA_BASE_PATH, HARDWARE_CLUSTER_MAPPING_FILE))
        # hardware_data.columns = [0, 1]
        # if hardware_cluster_to_be_used:
        #     hardware_data = hardware_data[hardware_data[1] in hardware_cluster_to_be_used]
        lang_data = read_txt_into_df(os.path.join(TRAINING_DATA_BASE_PATH, LANGUAGE_CLUSTER_MAPPING_FILE))
        lang_data.columns = [0, 1]
        if language_cluster_to_be_used:
            lang_data = lang_data[lang_data[1].isin(language_cluster_to_be_used)]
        lib_data = read_txt_into_df(os.path.join(TRAINING_DATA_BASE_PATH, LIBRARY_CLUSTER_MAPPING_FILE))
        lib_data.columns = [0, 1]
        if library_cluster_to_be_used:
            lib_data = lib_data[lib_data[1].isin(library_cluster_to_be_used)]
    return hardware_data, lang_data, lib_data


def write_words_to_text_file(path, list_of_words):
    with open(path, 'w+') as output:
        for sent in list_of_words:
            for word in sent:
                word = word.strip()
                if word:
                    output.write(word + ' O' + '\n')
            output.write('\n')


def main(
        hardware_cluster_to_be_used=[0],
        language_cluster_to_be_used=[0, 1],
        library_cluster_to_be_used=[0, 1],
        from_json=False,
        from_csv=True,
        from_text=False,
        hardware_output_path='/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson/extracted_sentences/annotated/hardware.text',
        lang_output_path='/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson/extracted_sentences/annotated/lang.text',
        lib_output_path='/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson/extracted_sentences/annotated/lib.text'
):
    spacy_processor = SpacyProcessor()
    hardware_data, lang_data, lib_data = get_data(from_csv, hardware_cluster_to_be_used, language_cluster_to_be_used,
                                                  library_cluster_to_be_used)
    # hardware_data = spacy_processor.get_words_from_text_sentences(hardware_data[0])
    lang_data = spacy_processor.get_words_from_text_sentences(lang_data[0].array[1:])
    lib_data = spacy_processor.get_words_from_text_sentences(lib_data[0].array[1:])
    # write_words_to_text_file(hardware_output_path, hardware_data)
    write_words_to_text_file(lang_output_path, lang_data)
    write_words_to_text_file(lib_output_path, lib_data)


if __name__ == '__main__':
    main()

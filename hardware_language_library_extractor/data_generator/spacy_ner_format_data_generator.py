import os

from hardware_language_library_extractor.common.util import get_regex_pattern, preprocessing_single_sentence, \
    read_txt_into_df, write_output_to_jsons
from hardware_language_library_extractor.data_generator.config import HARDWARE_KEYWORDS, MIN_THRESHOLD_SENTCHAR_LEN, \
    MAX_THRESHOLD_SENTCHAR_LEN, TRAINING_DATA_BASE_PATH, HARDWARE_CLUSTER_MAPPING_FILE, SPACY_NER_TRAIN_DATA

hardware_pattern = get_regex_pattern(HARDWARE_KEYWORDS)


def get_spacy_ner_data(data):
    train_data = []
    for sent in data:
        sent = preprocessing_single_sentence(sent, MIN_THRESHOLD_SENTCHAR_LEN, MAX_THRESHOLD_SENTCHAR_LEN)
        if sent:
            ent_matches = hardware_pattern.finditer(sent)
            ents = {"entities": []}
            for ent in ent_matches:
                ents["entities"].append((ent.start(), ent.end(), "hardware"))
            if ents["entities"]:
                train_data.append((sent, ents))
    return train_data


def main():
    data = read_txt_into_df(os.path.join(TRAINING_DATA_BASE_PATH, HARDWARE_CLUSTER_MAPPING_FILE))
    data.columns = [0, 1]
    data = data[data[1] == 0][0]
    train_data = get_spacy_ner_data(data)
    write_output_to_jsons(train_data, os.path.join(TRAINING_DATA_BASE_PATH, SPACY_NER_TRAIN_DATA))


if __name__ == '__main__':
    main()

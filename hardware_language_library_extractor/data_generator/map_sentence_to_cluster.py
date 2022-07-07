import pandas as pd
import json
import os
import numpy as np
from multiprocessing import set_start_method, Pool
from hardware_language_library_extractor.logger import Logger

logger = Logger('allenai_embeddings')
logger = logger.logger

base_path = '/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson/extracted_sentences/'
# base_path = '/home/group/cset/extracted_sentences/'
sentence_file_names = ['spacyhardware_filtered_sent.txt', 'spacylanguage_filtered_sent.txt',
                       'spacylibrary_filtered_sent.txt']
mapping_file_names = ['spacyhardwarecluster_labels_mapping.json', 'spacylanguagecluster_labels_mapping.json',
                      'spacylibrarycluster_labels_mapping.json']
cluster_numbers = ['4', '4', '4']


def load_sentences(file_name):
    df = pd.read_table(os.path.join(base_path, file_name), header=None)
    return df


def load_json_mapping(file_name):
    with open(os.path.join(base_path, file_name)) as inp:
        data = json.load(inp)
    return data


def merge_csv(file_name, df, mapping, key):
    try:
        df = df.iloc[:len(mapping[key])]
        df[1] = np.asarray(mapping[key])
        df.to_csv(os.path.join(base_path, '{}_sentence_mapping.csv'.format(file_name.split('_')[0])), encoding='utf-8',
                  index=False)
    except Exception as e:
        logger.error('Merging failed for file: {} and the error received is: {}'.format(file_name, e))


def driver(file_index):
    df = load_sentences(sentence_file_names[file_index])
    data = load_json_mapping(mapping_file_names[file_index])
    merge_csv(sentence_file_names[file_index], df, data, cluster_numbers[file_index])


def main():
    try:
        file_indices = [0, 1, 2]
        set_start_method('spawn')
        with Pool(processes=3) as pool:
            pool.map(driver, file_indices)
    except RuntimeError:
        pass


if __name__ == '__main__':
    main()

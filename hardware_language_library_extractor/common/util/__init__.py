import json
from typing import Pattern
import random
import os
import pandas as pd
import re

regex = re.compile(r'[^a-zA-Z\s]')


def load_data_from_json(path):
    with open(path, errors="ignore") as input_file:
        data = json.load(input_file)
    return data


def load_data_from_jsons(path):
    with open(path, errors="ignore") as input_file:
        data = json.loads(input_file.read())
    return data


def write_output_to_json(output, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(output, outfile, sort_keys=False, indent=4)


def write_output_to_jsons(output, path):
    with open(path, 'w', encoding='utf-8') as outfile:
        outfile.write(json.dumps(output))


def read_txt_into_df(path, header=None):
    df = pd.read_table(path, header=header)
    return df


def create_folder(path, name, recursive=False):
    try:
        if not recursive:
            os.mkdir(os.path.join(path, name))
        else:
            os.makedirs(os.path.join(path, name), exist_ok=True)
    except FileExistsError:
        print("Dir already exist")


def append_newline_to_text_file(text_to_append, path):
    with open(path, "a+") as file_object:
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        file_object.write(text_to_append)


def append_multiple_lines_to_text_file(lines_to_append, path):
    with open(path, "a+") as file_object:
        append_eol = False
        file_object.seek(0)
        data = file_object.read(100)
        if len(data) > 0:
            append_eol = True
        for line in lines_to_append:
            if append_eol:
                file_object.write("\n")
            else:
                append_eol = True
            file_object.write(line)


def pre_process_data(data: dict, max_len=150) -> dict:
    """
    @param data: It's a dict. Fork the key Sentences, it Contains List[List[Strings]]
    @return: Dict having same format as input param.
    """

    for i in range(len(data['sentences'])):
        data['sentences'][i] = [item for item in data['sentences'][i] if len(item) != 1 and regex.sub('', item)]
        if len(data['sentences'][i]) > max_len:
            data['sentences'][i] = data['sentences'][i][:max_len]
    return data


def preprocessing_single_sentence(sentence, min_threshold_sentchar_len, max_threshold_sentchar_len):
    sentence = regex.sub('', sentence).strip()
    if len(sentence) > min_threshold_sentchar_len and len(sentence) < max_threshold_sentchar_len:
        return sentence
    return None


def preprocessing_ssentences(ssentences, min_threshold_sentchar_len, max_threshold_sentchar_len):
    processed_sentences = []
    discarded_indexes = []
    for i in range(len(ssentences)):
        sent = ssentences[i]
        processed_sent = preprocessing_single_sentence(sent.text, min_threshold_sentchar_len, max_threshold_sentchar_len)
        if processed_sent:
            processed_sentences.append(processed_sent)
        else:
            discarded_indexes.append(i)
    return processed_sentences, discarded_indexes


def get_sentence_unit_skeleton(sentence, section_name, start_relative_section, end_relative_section, entities):
    sent_unit = dict()
    sent_unit['sentence'] = sentence
    sent_unit['section_name'] = section_name
    sent_unit['start_relative_section'] = start_relative_section
    sent_unit['end_relative_section'] = end_relative_section
    sent_unit['entities'] = entities
    return sent_unit


def get_intermediate_output_unit(positive_sent_indexes, sentences, category):
    output = dict()
    output[category] = []
    for ind in positive_sent_indexes:
        output[category].append(sentences[ind].text)
    return output


def get_output_skeleton(data):
    output = dict()
    output[data[0]["basename"]] = data[0]
    output['hardware_platforms'] = []
    output['language_libraries'] = []
    output['compute_resources'] = []
    return output


def get_regex_pattern(seed_words):
    pattern_string = ''
    for i in range(len(seed_words)):
        if i < len(seed_words) - 1:
            if len(seed_words[i]) == 1:
                pattern_string += '\\b{}\\b|'.format(seed_words[i])
            else:
                pattern_string += '\\b{}|'.format(seed_words[i])
        else:
            pattern_string += '\\b{}'.format(seed_words[i])
    pattern: Pattern[str] = re.compile(pattern_string, re.I)
    return pattern


def get_hardware_annotation_regex():
    pattern_string = '\\bCPU|\\bGPU|\\bTPU\\b|\\bNvidia|\\bTesla|\\bGEFORCE|\\bTITAN|\\bRTX\\b|\\bG-Sync|\\bCUDA' \
                     '|\\bNVENC\\b|\\bIntel\\b|\\bXeon|\\bPentium|\\bAPElink '
    pattern: Pattern[str] = re.compile(pattern_string, re.I)
    return pattern


def get_alias_name(paper_name):
    paper_name = paper_name.split('.')
    return ".".join(paper_name[:2])


def shuffle_list(sequence):
    random.shuffle(sequence)
    return sequence

import json
import pandas as pd
import random
import re

positive_data_path = '/home/group/cset/extracted_sentences/spacyhardware_sub_sent.txt'
negative_data_path = '/home/group/cset/extracted_sentences/negative_hardware_sents.txt'
train_file = '/home/group/cset/extracted_sentences/transformer/training.jsonl'
eval_file = '/home/group/cset/extracted_sentences/transformer/evaluation.jsonl'
MIN_THRESHOLD_SENT_LEN = 20
MAX_THRESHOLD_SENT_LEN = 150

regex = re.compile(r'[^a-zA-Z\s]')


def load_data(path):
    df = pd.read_table(path)
    df.columns = [0]
    return df


def write_to_jsonl(path, data):
    with open(path, 'w+') as output:
        for item in data:
            output.write(json.dumps(item) + '\n')


def preprocessing(sentences):
    processed = []
    for sent in sentences:
        sent = regex.sub('', sent).strip()
        if len(sent) > MIN_THRESHOLD_SENT_LEN:
            if len(sent) > MAX_THRESHOLD_SENT_LEN:
                sent = sent[:MAX_THRESHOLD_SENT_LEN]
            processed.append(sent)
    return processed


def main():
    positive_sents = preprocessing(load_data(positive_data_path)[0])
    negative_sents = preprocessing(load_data(negative_data_path)[0])
    positive = [(sent, {"cats": {"POSITIVE": 1.0, "NEGATIVE": 0.0}}) for sent in positive_sents]
    negative = [(sent, {"cats": {"POSITIVE": 0.0, "NEGATIVE": 1.0}}) for sent in negative_sents]
    data = positive + negative
    random.shuffle(data)
    write_to_jsonl(eval_file, data[-1000:])
    write_to_jsonl(train_file, data[:-1000])


if __name__ == '__main__':
    main()

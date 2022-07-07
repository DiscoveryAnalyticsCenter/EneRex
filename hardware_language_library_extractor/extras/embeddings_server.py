import numpy as np
import pandas as pd
import os
import torch
from transformers import *
from torch.multiprocessing import Pool, Process, set_start_method
from hardware_language_library_extractor.logger import Logger
import falcon
import json

logger = Logger('allenai_embeddings')
logger = logger.logger

base_path = '/mnt/c/Users/kkkoo/OneDrive/Documents/VirginiaTech/DACResearch/data/outputjson/outputjson' \
            '/extracted_sentences/ '
# base_path = '/home/group/cset/extracted_sentences/' file_names = ['spacyhardware_filtered_sent.txt',
# 'spacyhardware_sub_sent.txt', 'spacyhardware_prev_sent.txt', 'spacylanguage_filtered_sent.txt',
# 'spacylanguage_sub_sent.txt', 'spacylanguage_prev_sent.txt', 'spacylibrary_filtered_sent.txt',
# 'spacylibrary_sub_sent.txt', 'spacylibrary_prev_sent.txt']
file_names = ['spacylibrary_filtered_sent.txt']
batch_size = 10
max_len = 150

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
model.eval()


def get_embeddings(sentences):
    try:
        df = pd.DataFrame(sentences)

        tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
        padded = []
        for token in tokenized.values:
            token = token[:max_len]
            if len(token) < max_len:
                token = token + [0] * (max_len - len(token))
            padded.append(token)
        padded = np.array(padded)
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)
        features = last_hidden_states[0][:, 0, :].numpy()
        return features
    except Exception as e:
        print(e.args[0])
        logger.error('For sentences {} error received is: {}'.format(sentences, e), exc_info=True)


class VerifierResource():
    def on_post(self, req, resp):
        payload = json.loads(req.stream.read())
        embedding = get_embeddings(payload['data'])
        resp.body = json.dumps({"embedding": embedding.tolist()})
        resp.status = falcon.HTTP_200

    def on_get(self, req, resp):
        doc = {
            'images': [
                {
                    'href': '/images/1eaf6ef1-7f2d-4ecc-a8d5-6e8adba7cc0e.png'
                }
            ]
        }
        resp.body = json.dumps(doc, ensure_ascii=False)
        resp.status = falcon.HTTP_200


api = falcon.API()
api.add_route('/model', VerifierResource())

from transformers import *
import torch
from hardware_language_library_extractor import device


class Embeddings(object):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        self.model.eval()

    def get_embeddings(self, sentences):
        return self.model.encode(sentences)

    def get_embeddings_scierc(self, sentences):
        tokenized = self.tokenizer(sentences, truncation=True, max_length=150, padding=True, return_tensors='pt').to(
            device)
        with torch.no_grad():
            last_hidden_states = self.model(tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        return features

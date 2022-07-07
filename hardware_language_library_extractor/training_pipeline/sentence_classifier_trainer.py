import os
import logging
from typing import Dict, Optional

from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel, EvalPrediction
import torch
from transformers import (TrainingArguments, Trainer, set_seed, glue_compute_metrics)
from sklearn.model_selection import train_test_split
import numpy as np

from hardware_language_library_extractor.common.util import read_txt_into_df
from hardware_language_library_extractor.training_pipeline.config import TRAINING_DATA_BASE_PATH, \
    HARDWARE_POSITIVE_SENTENCES, HARDWARE_NEGATIVE_SENTENCES, LANGUAGE_POSITIVE_SENTENCES, LANGUAGE_NEGATIVE_SENTENCES,\
    LIBRARY_POSITIVE_SENTENCES, LIBRARY_NEGATIVE_SENTENCES, MAX_THRESHOLD_SENTCHAR_LEN, HARDWARE_SENTENCE_CLASSIFIER, \
    LANGUAGE_SENTENCE_CLASSIFIER, LIBRARY_SENTENCE_CLASSIFIER, OUTPUT_FOLDER_BASE_PATH, OVERWRITE_OUTPUT_DIRECTORY, \
    DO_TRAIN, DO_EVAL, PER_DEVICE_TRAIN_BATCH_SIZE, PER_DEVICE_EVAL_BATCH_SIZE, NUM_TRAIN_EPOCHS, LOGGING_STEPS, \
    LOGGING_FIRST_STEP, SAVE_STEPS, EVALUATE_DURING_TRAINING

logging.basicConfig(level=logging.INFO)


class SentenceDataset(Dataset):
    def __init__(self, sentences, targets, max_source_len, tokenizer):
        self.sentences = sentences
        self.targets = targets
        self.max_source_len = max_source_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sent = str(self.sentences[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=self.max_source_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation_strategy=True,
            truncation=True
        )
        return {'sentence': sent,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(target, dtype=torch.long)
                }


def load_data(path):
    df = read_txt_into_df(path)
    df.columns = [0]
    return df


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return glue_compute_metrics("mrpc", preds, p.label_ids)


def train_sentence_classifier(training_data_base_path, positive_sentences, negative_sentences, output_folder):
    random_seed = 42
    np.random.seed(random_seed)
    positive_sents_df = load_data(os.path.join(training_data_base_path, positive_sentences))
    negative_sents_df = load_data(os.path.join(training_data_base_path, negative_sentences))
    positive_sents_df[1] = [1] * len(positive_sents_df[0])
    negative_sents_df[1] = [0] * len(negative_sents_df)
    df_train, df_test = train_test_split(positive_sents_df.append(negative_sents_df), test_size=0.1,
                                         random_state=random_seed)
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    base_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased',
                                                               config=base_model.config)
    training_args = TrainingArguments(
        output_dir=output_folder,
        overwrite_output_dir=OVERWRITE_OUTPUT_DIRECTORY,
        do_train=DO_TRAIN,
        do_eval=DO_EVAL,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        logging_first_step=LOGGING_FIRST_STEP,
        save_steps=SAVE_STEPS,
        evaluate_during_training=EVALUATE_DURING_TRAINING,
    )
    set_seed(training_args.seed)
    train_dataset = SentenceDataset(df_train[0].to_numpy(), df_train[1].to_numpy(), MAX_THRESHOLD_SENTCHAR_LEN,
                                    tokenizer)
    test_dataset = SentenceDataset(df_test[0].to_numpy(), df_test[1].to_numpy(), MAX_THRESHOLD_SENTCHAR_LEN, tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


def main(
        train_hardware=True,
        train_language=True,
        train_library=True
):
    if train_hardware:
        train_sentence_classifier(TRAINING_DATA_BASE_PATH, HARDWARE_POSITIVE_SENTENCES, HARDWARE_NEGATIVE_SENTENCES,
                                  os.path.join(OUTPUT_FOLDER_BASE_PATH, HARDWARE_SENTENCE_CLASSIFIER))
    if train_language:
        train_sentence_classifier(TRAINING_DATA_BASE_PATH, LANGUAGE_POSITIVE_SENTENCES, LANGUAGE_NEGATIVE_SENTENCES,
                                  os.path.join(OUTPUT_FOLDER_BASE_PATH, LANGUAGE_SENTENCE_CLASSIFIER))
    if train_library:
        train_sentence_classifier(TRAINING_DATA_BASE_PATH, LIBRARY_POSITIVE_SENTENCES, LIBRARY_NEGATIVE_SENTENCES,
                                  os.path.join(OUTPUT_FOLDER_BASE_PATH, LIBRARY_SENTENCE_CLASSIFIER))


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.2.4
"""
from __future__ import unicode_literals, print_function

import plac
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

from hardware_language_library_extractor.common.util import *
from hardware_language_library_extractor.extras.config import *
import os

# training data
TRAIN_DATA = load_data_from_jsons(os.path.join(TRAINING_DATA_BASE_PATH, SPACY_NER_TRAIN_DATA))
LABEL = "hardware"


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(
        model=os.path.join(MODELS_FOLDER_BASE_PATH, TRANSFORMER_HARDWARE_CLASSIFIER),
        output_dir=Path(os.path.join(MODELS_FOLDER_BASE_PATH, NER_MODEL)),
        n_iter=10,
        batch_size=8,
        learn_rate=2e-5,
):
    """Load the model, set up the pipeline and train the entity recognizer."""
    # spacy.util.fix_random_seed(0)
    # is_using_gpu = spacy.prefer_gpu()
    # if is_using_gpu:
    #     torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        nlp_en = spacy.load('en_core_web_sm')
        ner = nlp_en.get_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("VEGETABLE")
    move_names = list(ner.move_names)
    # add labels
    # for _, annotations in TRAIN_DATA:
    #     for ent in annotations.get("entities"):
    #         ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            optimizer = nlp.begin_training()

        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.35,  # dropout - make it harder to memorise data
                    losses=losses
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA[:5]:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        assert nlp2.get_pipe("ner").move_names == move_names
        for text, _ in TRAIN_DATA[:5]:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

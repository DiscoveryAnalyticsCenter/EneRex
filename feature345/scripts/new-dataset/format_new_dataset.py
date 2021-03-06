"""
Format a new dataset.
"""

import argparse
import os
import json

import spacy


def format_document(fname, nlp):
    text = open(fname).read()
    doc = nlp(text)
    sentences = [[tok.text for tok in sent] for sent in doc.sents]
    #now check for anomaly, truncate sentences, larger than 300 tokens
    for sent in sentences:
        if len(sent) > 299:
            del sent[299:]
    doc_key = os.path.basename(fname).replace(".txt", "")
    res = {"doc_key": doc_key,
           "sentences": sentences}
    return res


def format_dataset(data_directory, output_file, use_scispacy):
    nlp_name = "en_core_sci_sm" if use_scispacy else "en_core_web_sm"
    nlp = spacy.load(nlp_name)

    fnames = [f"{data_directory}/{name}" for name in os.listdir(data_directory)]
    res = [format_document(fname, nlp) for fname in fnames]
    with open(output_file, "w") as f:
        for doc in res:
            print(json.dumps(doc), file=f)


def get_args():
    description = "Format an unlabled dataset, consisting of a directory of `.txt` files; one file per document."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("data_directory", type=str,
                        help="A directory with input `.txt files, one file per document.")
    parser.add_argument("output_file", type=str,
                        help="The output file, `.jsonl` extension recommended.")
    parser.add_argument("--use-scispacy", action="store_true",
                        help="If provided, use scispacy to do the tokenization.")
    return parser.parse_args()


def main():
    args = get_args()
    format_dataset(**vars(args))


if __name__ == "__main__":
    main()

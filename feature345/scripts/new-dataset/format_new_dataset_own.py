"""
Format a new dataset.
"""
import multiprocessing
from multiprocessing import Process, Value, Array
from multiprocessing.pool import Pool

import argparse
import os
import json
import glob
from pathlib import Path

import spacy


def format_document(fname):
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



def get_args():
    description = "Format an unlabled dataset, consisting of a directory of `.txt` files; one file per document."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("data_directory", type=str,
                        help="A directory with input `.txt files, one file per document.")
    parser.add_argument("output_file", type=str,
                        help="The output file, `.jsonl` extension recommended.")
    parser.add_argument("--use_scispacy", default=True,
                        help="If provided, use scispacy to do the tokenization.")
    parser.add_argument("--cleanDir", action='store_true',
                        help="If provided, clean the txtForDygie Directory")                    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    data_directory = args.data_directory
    output_file = args.output_file
    use_scispacy = args.use_scispacy
    cleanDir = args.cleanDir

    nlp_name = "en_core_sci_sm" if use_scispacy else "en_core_web_sm"
    print(nlp_name)
    nlp = spacy.load(nlp_name)

    # fnames = [f"{data_directory}/{name}" for name in os.listdir(data_directory)]
    # print(fnames)

    #only work with .txt, don't read .gitignore files
    fnames = sorted(Path.cwd().joinpath(data_directory).glob('*.txt'))

    pool = Pool()
    res = pool.map(format_document, fnames)
    
    
    with open(output_file, "w") as f:
        for doc in res:
            print(json.dumps(doc), file=f)

    if cleanDir:
        for item in fnames:
            os.remove(item)

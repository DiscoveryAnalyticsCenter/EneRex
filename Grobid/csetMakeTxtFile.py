import time
import re 
import string 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from tqdm import tqdm 
import networkx as nx
import copy
import os
from os import listdir
import json
import glob
from pathlib import Path
from os.path import isfile, join
import multiprocessing
from multiprocessing import Process, Value, Array
from multiprocessing.pool import Pool
import argparse
import nltk
from urllib.parse import urlparse
# import urltools

from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span 
from spacy import displacy 
from spacy.attrs import ORTH, NORM
from spacy.tokenizer import Tokenizer
import spacy.lang.char_classes
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex

import pickle
from itertools import islice

import en_core_web_sm
# pd.set_option('display.max_colwidth', 200)
# import textdistance
start_time = time.time()



######## NLP model load and some preliminary settings ################

nlp = en_core_web_sm.load()
#adding dataset stopwords
nlp.Defaults.stop_words |= {'data','database', 'dataset', 'corpus','corpora', 'data-set', 'data-base', 'datasets', 'databases'}

################### main interation###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselineDir", default="/home/raquib/SciIE/csetAnnotation/", help="path to the trainingData directory") 
    parser.add_argument("--output", default="/home/raquib/SLIE/dygiepp/inputData", help="path to the input directory") 
    args = parser.parse_args()

    baselineDir= args.baselineDir
    outpath = args.output
    

    basefiles = sorted(Path.cwd().joinpath(baselineDir).glob('*.jsonl'))
    data = []
    failed = []
    for item in basefiles:
        with open(item, 'r', encoding='utf-8') as f:
            #try to verify number of lines in each file, it its 500, except for the last file
            try:
                for line in f:
                    data.append(json.loads(line))
            except:
                failed.append(item)
    print("total file in baseline:", len(data), "failed to load:", len(failed))


    for item in data:
        fullString = item['title'] + ". " + item['abstract']

        if len(fullString) > 10:
            posixvalue = Path.cwd().joinpath(outpath, item["cset_id"])
            posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.txt')
            with open(posixvalue, 'w') as output:
                output.write(fullString)
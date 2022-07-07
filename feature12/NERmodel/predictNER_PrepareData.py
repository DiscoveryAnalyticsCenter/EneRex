import time
import re 
import string 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from tqdm import tqdm 
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

from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span 
from spacy import displacy 
from spacy.attrs import ORTH, NORM
from spacy.tokenizer import Tokenizer
import spacy.lang.char_classes
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

import pickle
from itertools import islice

import en_core_web_sm
# pd.set_option('display.max_colwidth', 200)
# import textdistance
start_time = time.time()


######## NLP model load and some preliminary settings ################
infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )

infix_re = compile_infix_regex(infixes)

nlp = en_core_web_sm.load()
nlp.tokenizer.infix_finditer = infix_re.finditer
#adding dataset stopwords, set added
nlp.Defaults.stop_words |= {'data','database', 'dataset', 'corpus','corpora', 'data-set', 'data-base', 'datasets', 'databases', 'set'}

##################################################################

def process(tupleValue):
    key = tupleValue[0]
    value = tupleValue[1]

    if NNlist is not None:
        if value and NNlist[key]:
            NNvalue = NNlist[key]
            
            TokensList = []
            for sent in value:
                #check all NN against this sentence
                sentence = sent['sentence']
                doc = nlp(sentence)

                #add to sentence at this point
                prevTokenIsEnt = False
                for tok in doc:
                    if not tok.is_stop and not tok.is_punct and not tok.is_space \
                    and any((tok.text in i) for i in NNvalue.keys()):
                        if prevTokenIsEnt:
                            TokensList.append(str(tok.text+ " " +"I-DAT"))
                        else:
                            TokensList.append(str(tok.text+ " " +"B-DAT"))

                        prevTokenIsEnt = True
                    elif not tok.is_punct and not tok.is_space:
                        TokensList.append(str(tok.text+ " " +"O"))
                        prevTokenIsEnt = False

                TokensList.append(str(""))

            TotalList.append(TokensList)

    else:
        if value:
            result = []
            resultwithMap = []
            
            for sent in value:
                #check all NN against this sentence
                sentence = sent['sentence']
                doc = nlp(sentence)
                TokensList = []
                TokensListwithMap = []

                #add to sentence at this point
                for tok in doc:
                    if not tok.is_punct and not tok.is_space:
                        TokensList.append(tok.text)
                        
                        # savingSpan.start_char-savingSpan.sent.start_char
                        start_span = tok.idx
                        end_span = tok.idx + len(tok)
                        stringWithSpan = str(tok.text + " " + str(start_span) + " " + str(end_span))
                        TokensListwithMap.append(stringWithSpan)

                TokensList.append(str(""))
                TokensListwithMap.append(str(""))

                result.append(TokensList)
                resultwithMap.append(TokensListwithMap)

            TotalDict[key] = result
            TotalDictWithMap[key] = resultwithMap


################### main interation###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentenceFile", default=None, help="path to the dataSentenceFile(in output folder of extraction algorithms)") 
    parser.add_argument("--datasetNameFile_FacetBased", default=None, help="path to the datasetNameFile_FacetBased(in output folder of facet-based extraction algorithms)") 
    parser.add_argument("--outputDir", default="NERmodel/data/", help="path to the output directory(data folder of NER model)") 
    parser.add_argument("--datasetNameFile_Given", action='store_true', help="indicate if datasetNameFile_Given,(only True in case when Training mode)") 
    parser.add_argument("--trainMode", action='store_true', help="indicate whether to create training data or test data(default false, will create test data). If yes, need datasetNameFile") 
    parser.add_argument("--n_core", default=40, help="number of CPU cores") 
    args = parser.parse_args()

    sentenceFile = args.sentenceFile
    datasetNameFile_FacetBased = args.datasetNameFile_FacetBased
    outputDir = args.outputDir
    datasetNameFile_Given = args.datasetNameFile_Given
    numberOfCore = int(args.n_core)
    trainMode = args.trainMode

    with open(sentenceFile, 'r') as f:
        mainlist = json.load(f)

    NNlist = None
    if datasetNameFile_Given:
        with open(datasetNameFile_FacetBased, 'r') as f:
            NNlist = json.load(f)

    # for key, value in islice(mainlist.items(), 0, 100):

    manager = multiprocessing.Manager()
    #dictionary needed for test data to keep track of which sentence belong to which file
    TotalDict = manager.dict()
    TotalDictWithMap = manager.dict()
    #Only list suffices for training mode
    TotalList = manager.list()
    pool = Pool(numberOfCore)
    pool.map(process, mainlist.items())


    if trainMode:
        first = math.floor(len(TotalList)*0.9)
        TRAIN_DATA = TotalList[0:first]
        DEV_DATA = TotalList[first:]
        print(len(TRAIN_DATA), len(DEV_DATA))

        train_data = []
        for item in TRAIN_DATA:
            train_data.extend(item)

        dev_data = []
        for item in DEV_DATA:
            dev_data.extend(item)

        with open(Path.cwd().joinpath(outputDir, 'train.txt.tmp'), "w") as outfile:
            outfile.write("\n".join(train_data))

        with open(Path.cwd().joinpath(outputDir, 'dev.txt.tmp'), "w") as outfile:
            outfile.write("\n".join(dev_data))

    else:
        test_data = []
        test_withmap = []

        for key, value in mainlist.items():
            if value:
                if key in TotalDict:
                    test_withmap.append(str("FileName:"+key))
                    for item in TotalDict[key]:
                        test_data.extend(item)
                    
                    for item in TotalDictWithMap[key]:
                        test_withmap.extend(item)

        with open(Path.cwd().joinpath(outputDir, 'test.txt'), "w") as outfile:
            outfile.write("\n".join(test_data))
        with open(Path.cwd().joinpath(outputDir, 'test_withmap.txt'), "w") as outfile:
            outfile.write("\n".join(test_withmap))

    print("--- %s seconds ---" % (time.time() - start_time))
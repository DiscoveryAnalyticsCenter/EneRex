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

from fuzzywuzzy import fuzz
import networkx as nx

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

# jsonFilesPath = '/home/group/cset/allCSjson'
# inpath = 'fullOutput'
# outpath = '/home/raquib/BertNER/data/'
# LABEL = "DATASET"

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
#adding dataset stopwords
nlp.Defaults.stop_words |= {'data','database', 'dataset', 'corpus','corpora', 'data-set', 'data-base', 'datasets', 'databases'}


################### main interation ###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentenceFile", help="path to the sentenceFile(in output folder of  extraction algorithms)") 
    parser.add_argument("--inputDir", help="path to the input directory(output folder of NER model)") 
    parser.add_argument("--outputDir", help="path to the input directory(output folder of NER model)") 
    args = parser.parse_args()

    sentenceFile = args.sentenceFile
    inputDir = args.inputDir
    outputDir = args.outputDir
    

    with open(sentenceFile, 'r') as f:
        mainlist = json.load(f)

    with open(Path.cwd().joinpath(inputDir, 'test_predictions.txt'), 'r') as f:
        predictions = f.readlines()

    

    indexList = [i for i, item in enumerate(predictions) if re.search(r"^FileName:.+$",item.strip())]
    # print(indexList)

    #issue here, solved
    #not reading the last file, last currentFile = predictions[indexList[i]:]
    #i should be run upto full length, if i is the last item, then the above equation
    #changing from range(0, len(indexList)-1) to range(0, len(indexList)) and adding if block

    for i in range(0, len(indexList)):
        if i == len(indexList)-1:
            currentFile = predictions[indexList[i]:]
        else:
            currentFile = predictions[indexList[i]:indexList[i+1]]
        # print(currentFile)
        key = currentFile[0][9:-1]
        # print(key)
        currentFile = currentFile[1:]
        if key in mainlist:
            separatorList = [i for i, item in enumerate(currentFile) if item == '\n']
            #len of separator gives number of sentences in this certain file
            #checking if output sentence count mismatch
            if len(separatorList) != len(mainlist[key]):
                # print("sentence Count mismatch")
                continue

            if len(separatorList) == len(mainlist[key]):
                for j in range(0, len(separatorList)):
                    #mainlist[key][j] is that sentence unit where you want to save these entities
                    if j == 0:
                        currentSent = currentFile[0:separatorList[j]]
                    else:
                        currentSent = currentFile[separatorList[j-1]:separatorList[j]]

                    entities = []
                    single_noun = []
                    spanList = []
                    for word in currentSent:
                        if "B-DAT" in word:
                            token = word[0:-7]

                            tokenStartSpan = int(token.split()[1])
                            tokenEndSpan = int(token.split()[2])
                            token = token.split()[0]

                            if len(token) <= 2 :
                                continue
                            entities.append(token)
                            single_noun.append(token)
                            spanList.append((tokenStartSpan, tokenEndSpan))

                        if "I-DAT" in word:
                            token = word[0:-7]

                            tokenStartSpan = int(token.split()[1])
                            tokenEndSpan = int(token.split()[2])
                            token = token.split()[0]

                            if len(token) <= 2 :
                                continue
                            if len(entities) >=1 :
                                entities[-1] = str(entities[-1]) + " " + str(token)
                                spanList[-1] = (spanList[-1][0], tokenEndSpan)
                            else:
                                entities.append(token)
                                spanList.append((tokenStartSpan, tokenEndSpan))

                            single_noun.append(token)
                        
                    mainlist[key][j]['entities_NER'] = entities
                    mainlist[key][j]['single_noun'] = single_noun
                    mainlist[key][j]['entity_spans'] = spanList
                        

    #deleting unnecessary keys from datasentences
    for key, value in mainlist.items():
        for item in value:
            #deleting unnecessary keys
            if "end_relative_section" in item:
                del item["end_relative_section"]
            if "end_seed" in item:
                del item["end_seed"]
            if "start_seed" in item:
                del item["start_seed"]
    
    # update sentence file
    with open(Path.cwd().joinpath(outputDir,'dataSentences.json'), 'w') as output:
        json.dump(mainlist, output, indent=4, sort_keys=True)



    #making NNResult file from mainlist. Will group it now
    NNResult = {}
    for key, value in mainlist.items():
        entitiesList = []
        for item in value:
            if "entities_NER" in item:
                entitiesList.extend(item["entities_NER"])
        NNResult[key] = list(set(entitiesList))

    newNNResult = {}
    # group them here
    for key, value in NNResult.items():
        G = nx.Graph()
        G.add_nodes_from(value)
        for x in value:
            for y in value:
                if x == y:
                    continue
                # print(x,y,fuzz.WRatio(x,y),fuzz.UWRatio(x,y),fuzz.ratio(x,y),fuzz.partial_ratio(x,y),fuzz.token_set_ratio(x,y))
                if fuzz.UWRatio(x,y) > 85:
                    G.add_edge(x, y)

        comps = [list(comp) for comp in nx.connected_components(G)]
        print(comps)
        newNNResult[key] = comps
        print()

                          
    with open(Path.cwd().joinpath(outputDir,'dataNNResult.json'), 'w') as output:
        json.dump(newNNResult, output, indent=4, sort_keys=True)
    
    # with open(Path.cwd().joinpath(outputDir,'dataNNResult.json'), 'w') as output:
    #     json.dump(NNResult, output, indent=4, sort_keys=True)

    print("--- %s seconds ---" % (time.time() - start_time))
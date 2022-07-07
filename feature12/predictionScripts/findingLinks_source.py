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
from string import punctuation

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

# jsonFilePath = '/home/group/cset/allCSjson'
# inpath = 'fullOutput'
# outpath = '/home/raquib/Extraction/findingLinks'

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

##################################################################

def process(tupleValue):
    key = tupleValue[0]
    value = tupleValue[1]
    listOfLinks = []

    if value:
        posixvalue = Path.cwd().joinpath(jsonFilePath, key)
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')
        with open(posixvalue, 'r') as f:
            jsonfile = json.load(f)
        
        #listOfLinks only hold context ones
        
        for sent in value:
            #only checking this sentence, context will be added later
            sentence = sent['sentence']
            doc = nlp(sentence)

            for tok in doc:
                if tok.like_url:
                    thisGroup = []
                    thisGroup.append(tok.text)
                    if not (tok.i+1) >= len(doc):
                        url = tok.text + tok.nbor().text
                        url = url.strip(punctuation)
                        if url != tok.text:
                            thisGroup.append(url)
                    if not (tok.i+2) >= len(doc):
                        url2 = tok.text + tok.nbor().text + tok.nbor().nbor().text
                        url2 = url2.strip(punctuation)
                        if url2 != url:
                            thisGroup.append(url2)
                    
                    listOfLinks.append(thisGroup)
                    
                    

            #match regex to find out the ref and footnotes
            allRefnFoots = re.findall(r"[#$]b[0-9]+",sentence)
            #without #b or $b tags, just a number
            extraFoots = re.findall(r"\s+[0-9]+\s+",sentence)
            allRefnFoots.extend(extraFoots)

            for item in allRefnFoots:
                # if '#' in item:
                #     if item[1:] in jsonfile[3]:
                #         referenceData = jsonfile[3][item[1:]]
                #         if referenceData[1]:
                #             listOfLinks.append((referenceData[0], referenceData[1]))
                #         if referenceData[3]:
                #             listOfLinks.append((referenceData[2], referenceData[3]))
                
                #also need to search ones without $ tags
                if '$' in item:
                    if item[2:] in jsonfile[2]:
                        footnote = jsonfile[2][item[2:]]
                        newDoc = nlp(footnote)
                        for tok in newDoc:
                            if tok.like_url:
                                thisGroup = []
                                thisGroup.append((tok.text))
                                listOfLinks.append(thisGroup)

                else:
                    if item in jsonfile[2]:
                        footnote = jsonfile[2][item]
                        newDoc = nlp(footnote)
                        for tok in newDoc:
                            if tok.like_url:
                                thisGroup = []
                                thisGroup.append((tok.text))
                                listOfLinks.append(thisGroup)

    TotalDict[key] = listOfLinks

        

################### main interation###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/raquib/SLIE/output/transformer/", help="path to the input directory") 
    parser.add_argument("--output", default="/home/raquib/SLIE/output/transformer/", help="path to the output directory") 
    parser.add_argument("--n_core", default=80, help="number of CPU cores")
    parser.add_argument("--jsonFilePath", default='/home/raquib/SLIE/data/JSONs/', help="path to the JSON files")  
    args = parser.parse_args()

    inpath = args.input
    outpath = args.output
    numberOfCore = int(args.n_core)
    jsonFilePath = Path(args.jsonFilePath).resolve() 

    with open(Path.cwd().joinpath(inpath, 'sourceSentences.json'), 'r') as f:
        mainlist = json.load(f)

    
    # for key, value in islice(mainlist.items(), 0, 100):
    #first pass
    manager = multiprocessing.Manager()
    TotalDict = manager.dict()
    pool = Pool(numberOfCore)
    pool.map(process, mainlist.items())


    with open(Path.cwd().joinpath(outpath,'sourceFilteredLink.json'), 'w') as output:
        json.dump(TotalDict._getvalue(), output, indent=4, sort_keys=True)

    print("--- %s seconds ---" % (time.time() - start_time))
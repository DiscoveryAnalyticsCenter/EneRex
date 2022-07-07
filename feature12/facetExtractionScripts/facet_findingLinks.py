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
    listOfLinks2 = []

    if value:
        NNvalue = list(NNlist[key].keys())
        posixvalue = Path.cwd().joinpath(jsonFilesPath, key)
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')
        with open(posixvalue, 'r') as f:
            jsonfile = json.load(f)
        
        #listOfLinks only hold context ones
        
        dataSetNamesNN= []
        for sent in value:
            #only checking this sentence, context will be added later
            sentence = sent['sentence']
            dataSetNamesNN.extend(list(sent["dataset_name_candidates"].keys()))
            doc = nlp(sentence)

            for tok in doc:
                if tok.like_url:
                    listOfLinks.append(("URL", tok.text))

            #match regex to find out the ref and footnotes
            allRefnFoots = re.findall(r"[#$]b[0-9]+",sentence)
            #without #b or $b tags, just a number
            extraFoots = re.findall(r"\s+[0-9]+\s+",sentence)
            allRefnFoots.extend(extraFoots)

            for item in allRefnFoots:
                if '#' in item:
                    if item[1:] in jsonfile[3]:
                        referenceData = jsonfile[3][item[1:]]
                        if referenceData[0]:
                            listOfLinks.append((referenceData[0], referenceData[1]))
                        elif referenceData[2]:
                            listOfLinks.append((referenceData[2], referenceData[3]))
                
                #also need to search ones without $ tags
                elif '$' in item:
                    if item[2:] in jsonfile[2]:
                        footnote = jsonfile[2][item[2:]]
                        newDoc = nlp(footnote)
                        for tok in newDoc:
                            if tok.like_url:
                                listOfLinks.append((footnote, tok.text))

                else:
                    if item in jsonfile[2]:
                        footnote = jsonfile[2][item]
                        newDoc = nlp(footnote)
                        for tok in newDoc:
                            if tok.like_url:
                                listOfLinks.append((footnote, tok.text))


        if NNvalue:
            #not necessary rn
            for i in range(0, len(NNvalue)):
                nn = NNvalue[i]
                newNN = [word for word in re.split("\W+",nn) if word.lower() not in nlp.Defaults.stop_words]
                NNvalue[i] = " ".join(newNN)
            
            #make a list of valid NN dataset name candidates and use those in subsequent matches
            dataSetNamesNN = set(dataSetNamesNN)
            selectedNN = []
            for name in dataSetNamesNN:
                if any((name in k or name.lower() in k) for k in NNvalue):
                    selectedNN.append(name)

            for item in listOfLinks:
                matched = False
                for nn in selectedNN:
                    if nn in item[0] or nn in item[1]:
                        listOfLinks2.append((nn, item[0], item[1]))

            #listofLinks2 still empty, so search for each NNvalue in all text
            if not listOfLinks2:
                for nn in selectedNN:
                    for value in jsonfile[2].values():
                        if nn in value:
                            newDoc = nlp(value)
                            for tok in newDoc:
                                if tok.like_url:
                                    listOfLinks2.append((value, tok.text))



                    # I don't need this part, without link and context
                    #checking for links should verify this part
                    for referenceData in jsonfile[3].values():
                        if referenceData[1] != "" and (nn in referenceData[0] or nn in referenceData[1]):
                            listOfLinks2.append((referenceData[0], referenceData[1]))
                        if referenceData[3] != "" and (nn in referenceData[2] or nn in referenceData[3]):
                            listOfLinks2.append((referenceData[2], referenceData[3]))
            
            #search for parent paper and looks for the 
            # if not listOfLinks2:



    TotalDict[key] = listOfLinks
    FinalDict[key] = listOfLinks2

        

def secondPass(tupleValue):
    key = tupleValue[0]
    value = tupleValue[1]
    linksOfSecondPass = []

    if value:
        NNvalue = list(NNlist[key].keys())
        posixvalue = Path.cwd().joinpath(jsonFilesPath, key)
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')
        with open(posixvalue, 'r') as f:
            jsonfile = json.load(f)
        
        
        dataSetNamesNN= []
        for sent in value:
            #only checking this sentence, context will be added later
            sentence = sent['sentence']
            dataSetNamesNN.extend(list(sent["dataset_name_candidates"].keys()))
            
        if NNvalue:
            #make a list of valid NN dataset name candidates and use those in subsequent matches
            dataSetNamesNN = set(dataSetNamesNN)
            selectedNN = []
            for name in dataSetNamesNN:
                if any((name in k or name.lower() in k) for k in NNvalue):
                    selectedNN.append(name)

            #second passing over content of already filtered links
            for outerItem in FinalDict.values():
                for item in outerItem:
                    matched = False
                    for nn in selectedNN:
                        if len(item) == 2 and item[1] != "" and (nn in item[0] or nn in item[1]):
                            linksOfSecondPass.append((nn, item[0], item[1]))
                        elif len(item) == 3 and item[2] != "" and (nn in item[0] or nn in item[1] or nn in item[2]):
                            linksOfSecondPass.append((nn, item[0], item[1]))


    
    FinalDict[key].extend(linksOfSecondPass)


################### main interation###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/group/cset/output/facetBased", help="path to the input directory") 
    parser.add_argument("--output", default="/home/group/cset/output/facetBased", help="path to the output directory") 
    parser.add_argument("--n_core", default=80, help="number of CPU cores")
    parser.add_argument("--jsonFilesPath", default='/home/raquib/SLIE/data/JSONs/', help="path to the JSON files")  
    args = parser.parse_args()

    inpath = args.input
    outpath = args.output
    numberOfCore = int(args.n_core)
    jsonFilesPath = args.jsonFilesPath

    with open(Path.cwd().joinpath(inpath, 'dataSentences.json'), 'r') as f:
        mainlist = json.load(f)

    with open(Path.cwd().joinpath(inpath, 'dataNNResult.json'), 'r') as f:
        NNlist = json.load(f)

    # for key, value in islice(mainlist.items(), 0, 100):
    #first pass
    manager = multiprocessing.Manager()
    TotalDict = manager.dict()
    FinalDict = manager.dict()
    pool = Pool(numberOfCore)
    pool.map(process, mainlist.items())

    #second pass
    pool = Pool(numberOfCore)
    pool.map(secondPass, mainlist.items())

    with open(Path.cwd().joinpath(outpath,'dataFilteredLink.json'), 'w') as output:
        json.dump(FinalDict._getvalue(), output, indent=4)

    print("--- %s seconds ---" % (time.time() - start_time))
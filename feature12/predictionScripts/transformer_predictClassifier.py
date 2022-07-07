import time
import spacy 
import pandas as pd 
import numpy as np 
import json
import glob
from pathlib import Path
import argparse
import os
from numpy import savez_compressed, load
import joblib

#transformer imports
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

# from utils_data import SciDataTrainingArguments as DataTrainingArguments
# from utils_data import SciDataset

from utils_data import SciDataTrainingArguments as DataTrainingArguments
from utils_data import SciDataset
import csv


from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span 
from spacy import displacy 
from spacy.attrs import ORTH, NORM
from spacy.tokenizer import Tokenizer
import spacy.lang.char_classes
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

import en_core_web_sm
# pd.set_option('display.max_colwidth', 200)
# import textdistance

import torch
from transformers import *
import torch.multiprocessing
from torch.multiprocessing import Pool, Process, set_start_method
# set_start_method("spawn", force=True)

import matplotlib.pyplot as plt
import matplotlib.cm as cm

start_time = time.time()

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

nlp.disable_pipes('ner')
def spacy_lemmatizer(text, nlp):
    
    docs=[]
    for i, item in enumerate(text):
        doc = nlp(item)
        docs.append(' '.join([tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]))
        
    return docs

################################

with open('modelLocation.json', 'r') as f:
    modelLocation = json.load(f)

tempDataDir = modelLocation["tempDataDir"]
model_source_path = modelLocation["model_source_path"]
dataset_model_path = modelLocation["dataset_model_path"]


set_seed(42)
num_labels = 2
output_mode = "classification"

dataArgs = DataTrainingArguments()
trainingArgs = TrainingArguments(tempDataDir)

#source model
source_config = AutoConfig.from_pretrained(
    model_source_path,
    num_labels=num_labels,
)
source_tokenizer = AutoTokenizer.from_pretrained(
    model_source_path,
)
source_model = AutoModelForSequenceClassification.from_pretrained(
    model_source_path,
    config=source_config,
)

source_trainer = Trainer(
    model=source_model,
    args=trainingArgs,
    train_dataset=None,
    eval_dataset=None,
)

#dataset model
dataset_config = AutoConfig.from_pretrained(
    dataset_model_path,
    num_labels=num_labels,
)
dataset_tokenizer = AutoTokenizer.from_pretrained(
    dataset_model_path,
)
dataset_model = AutoModelForSequenceClassification.from_pretrained(
    dataset_model_path,
    config=dataset_config,
)

dataset_trainer = Trainer(
    model=dataset_model,
    args=trainingArgs,
    train_dataset=None,
    eval_dataset=None,
)




def process(file):
    # print("inside process of pool")

    with open(file, 'r') as f:
        mainlist = json.load(f)

    # print("loaded file", mainlist[0]['basename'])

    datasetCandSentences = []
    sourceCodeSentences = []
        
    for sectionKey, value in mainlist[1].items():
        ########## common for both ######################
        # start = time.time()
        doc=nlp(value)
        sentences = [sent.text for sent in doc.sents]

        #checking if it is an empty section
        if len(sentences) <= 0:
            continue

        # sentences = spacy_lemmatizer(sentences, nlp)
        sentences = [[i] for i in sentences]
        # print("sentence count in section", len(sentences))
        
        #finally saving the results
        listOfSents = list(doc.sents)



        #making SciDataset for dataset part with dataset_tokenizer
        test_dataset = SciDataset(args = dataArgs, listOfString = sentences, tokenizer=dataset_tokenizer, mode="test")
    
        ########### dataset model #####################
        #dataset model trainer prediction
        predictions = dataset_trainer.predict(test_dataset=test_dataset).predictions
        dataLabels = np.argmax(predictions, axis=1)
        # print("dataset test prediction:")
        # print(dataLabels)

        #follow comment in sourcecode
        if len(dataLabels) == len(listOfSents):
            # print("length of sentence in doc and length of datalabels mismatched")

            for i in range(0, len(dataLabels)):
                if dataLabels[i] == 1:
                    newdict={}
                    newdict['sentence']=listOfSents[i].text

                    if i >= 1:
                        newdict['prev_sent'] = listOfSents[i-1].text

                    if i+1 < len(dataLabels):
                        newdict['next_sent'] = listOfSents[i+1].text

                    #saving the span information for website showing tupleSpan[0]:tupleSpan[1]]
                    savingSpan = listOfSents[i]
                    newdict['section_name']=sectionKey
                    newdict['start_relative_section']=savingSpan.start_char
                    # newdict['end_relative_section']=savingSpan.end_char
                    # newdict['start_seed']=savingSpan.start_char-savingSpan.sent.start_char
                    # newdict['end_seed']=savingSpan.end_char-savingSpan.sent.start_char

                    datasetCandSentences.append(newdict)


        ############# source model ##########################
        #making SciDataset for source with source_tokenizer
        test_dataset = SciDataset(args = dataArgs, listOfString = sentences, tokenizer=source_tokenizer, mode="test")
    
        predictions = source_trainer.predict(test_dataset=test_dataset).predictions
        sourceLabels = np.argmax(predictions, axis=1)
        

        # print("length of sentence in doc and length of sourcelabels mismatched")
        #need to check if the predicted length and list of sent length is equal, otherwise, data missing, continue
        # for loop is moved inside if block
        if len(sourceLabels) == len(listOfSents):
            # print("length of sentence in doc and length of sourcelabels mismatched")

            for i in range(0, len(sourceLabels)):
                if sourceLabels[i] == 1:
                    newdict={}
                    newdict['sentence']=listOfSents[i].text

                    if i >= 1:
                        newdict['prev_sent'] = listOfSents[i-1].text

                    if i+1 < len(sourceLabels):
                        newdict['next_sent'] = listOfSents[i+1].text

                    #saving the span information for website showing tupleSpan[0]:tupleSpan[1]]
                    savingSpan = listOfSents[i]
                    newdict['section_name']=sectionKey
                    newdict['start_relative_section']=savingSpan.start_char
                    # newdict['end_relative_section']=savingSpan.end_char
                    # newdict['start_seed']=savingSpan.start_char-savingSpan.sent.start_char
                    # newdict['end_seed']=savingSpan.end_char-savingSpan.sent.start_char

                    sourceCodeSentences.append(newdict)
                   
    result = {}
    result['basename'] = mainlist[0]['basename']
    result['dataset'] = datasetCandSentences
    result['sourceCode'] = sourceCodeSentences
    posixvalue = Path.cwd().joinpath(tempDataDir, mainlist[0]['basename'])
    posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')

    with open(posixvalue, 'w') as output:
        json.dump(result, output, indent=4)
        
    


################### main interation###################
######################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonFilePath", default="/home/raquib/SLIE/data/JSONs/", help="path to the input directory (JSON file that you want to extract information from)") 
    parser.add_argument("--output", default="/home/raquib/SLIE/output/transformer/", help="path to the output directory (output/transformer)") 
    parser.add_argument("--SourceInActive", action='store_false', help="indicate if Source Code Extraction is inactive (By default both is active)") 
    parser.add_argument("--DataInActive", action='store_false', help="indicate if Dataset Extraction is inactive (By default both is active)") 
    parser.add_argument("--n_core", default=1, help="number of CPU cores (Default 1, Using multiple CPU would require lots of CUDA memory)") 
    parser.add_argument("--debugMode", action='store_true', help="debug mode, limit number of processed file to 5") 
    args = parser.parse_args()

    
    jsonFilePath = Path(args.jsonFilePath).resolve() 
    outpath = args.output
    numberOfCore = int(args.n_core)

    SourceActive = args.SourceInActive
    DataActive = args.DataInActive
    debugMode = args.debugMode

    print("Classifier for Extraction with the following parameters")
    print("jsonFilePath:", jsonFilePath)
    print("output:", outpath)
    print("Number of Core using", numberOfCore)
    print("SourceActive", SourceActive, "DataActive", DataActive)
        
    print("reading file now")
    selectedFiles = sorted(Path.cwd().joinpath(jsonFilePath).glob('*.json'))
    print("number of files to process", len(selectedFiles))

    if debugMode:
        selectedFiles = selectedFiles[0:5]
        print("debug mode: only processing 5 files")

    try:
        set_start_method('spawn', force=True)
        pool = Pool(numberOfCore)
        pool.map(process, selectedFiles)
    except Exception as exception:
        print("EXCEPTION(): ", exception)

    print("out of pools, saving data now")

    
    # compiling all files now, file deleting part
    outputfiles = sorted(Path.cwd().joinpath(tempDataDir).glob('*.json'))
    dataFinalResult = {}
    sourceFinalResult = {}
    for item in outputfiles:
        with open(item, 'r') as f:
            itemData = json.load(f)
        
        dataFinalResult[itemData['basename']] = itemData['dataset']
        sourceFinalResult[itemData['basename']] = itemData['sourceCode']
        #removing this file
        os.remove(item)


    with open(Path.cwd().joinpath(outpath,'dataSentences.json'), 'w') as output:
        json.dump(dataFinalResult, output, indent=4)

    with open(Path.cwd().joinpath(outpath,'sourceSentences.json'), 'w') as output:
        json.dump(sourceFinalResult, output, indent=4)
        

    print("--- %s seconds ---" % (time.time() - start_time))
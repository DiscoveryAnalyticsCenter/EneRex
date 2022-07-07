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


from sklearn.utils import shuffle

import en_core_web_sm

start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingDataDir", default="/home/raquib/SLIE/TrainingDir", help="path to the trainingData directory") 
    parser.add_argument("--datasetMode", default=True, help="indicate if processing dataset feature (Only one True at a time)") 
    parser.add_argument("--sourceCodeMode", default=False, help="indicate if processing source code feature (Only one True at a time)") 
    parser.add_argument("--outpath", default="/home/raquib/SLIE/classifierTransformer/noncontext/datasetData", help="path to the trainingData directory (Different directory for dataset feature and source code feature, change accordingly)") 
    parser.add_argument("--contextMode", action='store_true', help="indicate if Unlabeled data come from context (Default False, so include all text)") 
    args = parser.parse_args()

    trainingDataDir = args.trainingDataDir
    datasetMode = args.datasetMode
    sourceCodeMode = args.sourceCodeMode
    outpath = args.outpath
    contextMode = args.contextMode
    
    print("trainingData Directory:", trainingDataDir)
    

    
    if datasetMode:
        correctfilename = 'dataCorrectSents.txt'
        incorrectfilename = 'dataIncorrectSents.txt'
    elif sourceCodeMode:
        correctfilename = 'sourceCorrectSents.txt'
        incorrectfilename = 'sourceIncorrectSents.txt'
    
    
    with open(Path.cwd().joinpath(trainingDataDir, correctfilename), 'r') as f:
        Labeled = [line.rstrip() for line in f]

    with open(Path.cwd().joinpath(trainingDataDir, incorrectfilename), 'r') as f:
        UNLabeled = [line.rstrip() for line in f]

    if not contextMode:
        sizeOfUnLabeled = len(Labeled)*2
    else:
        sizeOfUnLabeled = len(UNLabeled)

    print("incorrect dataset size", sizeOfUnLabeled)

    fullText = []
    fullText.extend(Labeled)
    #taking twice data from Unlabeled data
    fullText.extend(UNLabeled[0:sizeOfUnLabeled])

    correct = np.ones((len(Labeled), ), dtype=int)
    #taking twice data from Unlabeled data
    incorrect = np.zeros((sizeOfUnLabeled, ), dtype=int)
    totalY = np.concatenate((correct, incorrect), axis=None)

    #########################################
    df = pd.DataFrame({'label': totalY, 'text':fullText})
    df = shuffle(df, random_state=42)
    print("dataframe shape:", df.shape)
    print(df.head)

    secondDF = df.reindex(columns = ['label', 'blank', 'text'])
    print(secondDF.head)
    print(df.head)


    first = math.floor(len(secondDF)*0.85)
    second = math.floor(len(secondDF)*0.95)
       

    secondDF[:first].to_csv(Path.cwd().joinpath(outpath, 'train.tsv'), sep = '\t',  header=False)
    secondDF[first:second].to_csv(Path.cwd().joinpath(outpath, 'dev.tsv'), sep = '\t',  header=False)
    df[second:].to_csv(Path.cwd().joinpath(outpath, 'test.tsv'), sep = '\t',  header=False, index = False)


    print("--- %s seconds ---" % (time.time() - start_time))

   
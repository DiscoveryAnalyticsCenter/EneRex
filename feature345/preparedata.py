import time
import string 
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


start_time = time.time()

def process(file):
    with open(file, 'r') as f:
        mainlist = json.load(f)

    fullString = ""

    if includeAllSec:
        for key, value in mainlist[1].items():
            fullString = fullString + value + ". "

    elif onlyAbstract:
        selectedSection = ['title', 'abstract']
        for key, value in mainlist[1].items():
            if any(word in key.lower() for word in selectedSection):
                fullString = fullString + value + ". "

    else:
        selectedSection = ['title', 'abstract', 'introduction', 'conclusion', 'discussion', 'concluding', "remark"]
        for key, value in mainlist[1].items():
            if any(word in key.lower() for word in selectedSection):
                fullString = fullString + value + ". "
    


    if len(fullString) > 10:
        posixvalue = Path.cwd().joinpath(outpath, mainlist[0]['basename'])
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.txt')
        with open(posixvalue, 'w') as output:
            output.write(fullString)


################### main###################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="path to the input directory") 
    parser.add_argument("--output", default=None, help="path to the output directory(where .txt files will be created for dygiee system)") 
    parser.add_argument("--n_core", default=40, help="number of CPU cores") 
    parser.add_argument("--includeAllSec", action = 'store_true', help="To include all sections. By default these are included: title, abstract, introduction, conclusion, discussion, remarks") 
    parser.add_argument("--onlyAbstract", action = 'store_true', help="To extract only title, abstract. By default these are included: title, abstract, introduction, conclusion, discussion, remarks") 
    
    args = parser.parse_args()

    inpath = Path(args.input).resolve()
    outpath = args.output
    numberOfCore = int(args.n_core)
    includeAllSec = args.includeAllSec
    onlyAbstract = args.onlyAbstract
    
    selectedFiles = []
    selectedFiles = sorted(Path.cwd().joinpath(inpath).glob('*.json'))

    print("number of files to process", len(selectedFiles))
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    pool = Pool(numberOfCore)
    pool.map(process, selectedFiles)


    print("--- %s seconds ---" % (time.time() - start_time))
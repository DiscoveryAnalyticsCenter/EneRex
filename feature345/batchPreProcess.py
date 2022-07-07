import time
import pandas as pd 
import numpy as np 
import os
import json
from pathlib import Path
from os.path import isfile, join
import argparse

start_time = time.time()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonlFile", default=None, help="path to the input directory") 
    parser.add_argument("--batchDir", default=None, help="path to the input directory") 
    parser.add_argument("--batchOutputDir", default=None, help="path to the input directory") 
    parser.add_argument("--mergedResultFile", default=None, help="path to the input directory") 
    parser.add_argument("--preprocess", action = 'store_true', help="Preprocess mode or not") 
    parser.add_argument("--batchSize", default = 500, help="Preprocess mode or not") 
    args = parser.parse_args()
    
    batchDir= args.batchDir
    jsonlFile = args.jsonlFile
    preprocess = args.preprocess
    batchOutputDir = args.batchOutputDir
    mergedResultFile = args.mergedResultFile
    batchSize = int(args.batchSize)

    if preprocess:
        print("Pre-process mode")
        data = []
        with open(jsonlFile, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
            
        print(len(data))
        #now check for anomaly
        for item in data:
            for sent in item['sentences']:
                if len(sent) > 299:
                    del sent[299:]

        #check again if length is okay
        lengthOkay = True
        for item in data:
            if max([len(sent) for sent in item['sentences']]) > 299:
                lengthOkay = False
        print("length Okay status", lengthOkay)

        for i in range(0,len(data),batchSize):
            filename = str(i)+".jsonl"
            with open(Path.cwd().joinpath(batchDir, filename), "w") as f:
                for doc in data[i:i+batchSize]:
                    print(json.dumps(doc), file=f)

    else:
        print("Post-process mode")
        files = sorted(Path.cwd().joinpath(batchOutputDir).glob('*.jsonl'))

        data = []
        for item in files:
            with open(item, 'r', encoding='utf-8') as f:

                for line in f:
                    data.append(line[0:-1])


        with open(Path.cwd().joinpath(mergedResultFile), "w") as f:
            for doc in data:
                print(doc, file=f)

    print("--- %s seconds ---" % (time.time() - start_time))



    

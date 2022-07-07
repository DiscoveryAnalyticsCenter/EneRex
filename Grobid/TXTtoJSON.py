import time
import copy
import json
import glob
from pathlib import Path
import argparse
import multiprocessing
from multiprocessing.pool import Pool
import os
from os.path import join
import re
start_time = time.time()

    
def createJson(file, basename):
    with open(file, 'r') as reader:
        data = reader.readlines()
        
    
    fulltext={}
    fulltext['title']=data[0][:-1]
    fulltext['abstract']=data[1]

    footnote={}
    reference={}
    table={}
    
    report={}
    report['basename']=basename
    #how many sections in here, only title and abstract, so 2
    report['fulltext']=2
    #how many footnotes
    report['footnote_size']=0
    #how many references
    report['reference']=0
    #authors' last names
    report['authors'] = []
    
    result=[]
    result.append(report)
    result.append(fulltext)
    result.append(footnote)
    result.append(reference)
    result.append(table)

    return result

def process(file):
    basename=os.path.basename(file)
    if basename.endswith('txt'):
        basename= basename[0:-4]
    
    posixvalue = Path.cwd().joinpath(basename)
    posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')

    if not Path(posixvalue).is_file() or overwrite:
        result = createJson(file, basename)

        with open(posixvalue, 'w') as output:
            json.dump(result, output, indent=4)
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="path to the input directory(TXTs)") 
    parser.add_argument("--output", default=None, help="path to the output directory(JSONs)") 
    parser.add_argument("--numberOfCore", default=40, help="number of core") 
    parser.add_argument("--overwrite", action='store_true', help="Overwrite the output directory") 
    args = parser.parse_args()

    inpath = Path(args.input).resolve() 
    outpath = Path(args.output).resolve()
    overwrite = args.overwrite
    numberOfCore = int(args.numberOfCore)

    print("Txt to JSON generator with the following parameter")
    print("input directory:", inpath)
    print("output directory:", outpath)
    print("overwrite activated:", overwrite)
    print("number of CPU used:", numberOfCore)
    

    files = sorted(Path.cwd().joinpath(inpath).glob('*.txt'))
    os.chdir(Path.cwd().joinpath(outpath)) 
    pool = Pool(numberOfCore)
    pool.map(process, files)

    print("--- %s seconds ---" % (time.time() - start_time))
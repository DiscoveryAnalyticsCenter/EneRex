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
import shutil

start_time = time.time()

    
def createJson(data):
    fulltext={}
    fulltext['title']=data['title']
    fulltext['abstract']=data['abstract']

    footnote={}
    reference={}
    table={}
    
    report={}
    report['basename']=data['cset_id']
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

def process(file, outdir):
    result = []
    for line in open(file):
        record = createJson(json.loads(line))
        result.append(record)
    
    # need to get the file's basename for writing to output directory
    filename = os.path.basename(file)

    with open(os.path.join(outdir, f'{filename}'), 'w') as f:
        for item in result:
            f.write(json.dumps(item) + "\n")
        
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="path to the input directory(CSET format)") 
    parser.add_argument("--output", default=None, help="path to the output directory(JSONs)") 
    parser.add_argument("--numberOfCore", default=40, help="number of core") 
    parser.add_argument("--overwrite", action='store_true', help="Overwrite the output directory") 
    args = parser.parse_args()

    inpath = Path(args.input).resolve() 
    outpath = Path(args.output).resolve()
    overwrite = args.overwrite
    numberOfCore = int(args.numberOfCore)

    print("CSET format files to JSON generator with the following parameter")
    print("input directory:", inpath)
    print("output directory:", outpath)
    print("overwrite activated:", overwrite)
    print("number of CPU used:", numberOfCore)

    if len(os.listdir(args.output)) > 0 and not args.overwrite:
        raise ValueError(f"output directory {args.output} is not empty. Pass the --overwrite argument to overwrite")
    if args.overwrite:
        assert args.output not in ["/", "\\", "."]
        shutil.rmtree(args.output)
        os.mkdir(args.output)
    
    files = sorted(Path.cwd().joinpath(inpath).glob('*.jsonl'))
    with Pool(numberOfCore) as pool:
        pool.starmap(process, [(fi, args.output) for fi in files])

    print("--- %s seconds ---" % (time.time() - start_time))

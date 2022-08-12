import argparse
import json
import collections
from pathlib import Path
import os

# import multiprocessing
# from multiprocessing import Process, Value, Array
# from multiprocessing.pool import Pool

import random
from shutil import copy2



def convert(data):
    ret = ""
    
    body = data[1]
    for key, value in body.items():
        if key == "title":
            ret += value + "\n" + "\n"
        elif key == "abstract":
            ret += "Abstract" + "\n" + value + "\n" + "\n"
        else:
            ret += key + "\n" + value + "\n" + "\n"

    footNote = data[2]
    if footNote:
        ret += "Footnote" + "\n"
        for key, value in footNote.items():
            ret += key + " : " + value + "\n"



    return ret.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    outdir = parser.add_argument("--outdir", default="./file_list")

    options = parser.parse_args()
    outdir = Path(options.outdir)
    if not outdir.exists():
        outdir.mkdir()

    
    data = []
    with open("arxiv-ai-sample.jsonl", 'r') as f:
        for line in f:
            data.append(json.loads(line))
        
    print(len(data))
    data = sorted(data, key=lambda d: d['id'], reverse=True) 
    data = [d['id'] for d in data]
    
    print(data[0])
    print(data[1])

    count = 0

    existingData = []
    for item in data:
        posixvalue = Path.cwd().joinpath("/home/group/cset/newAllCSjson", item)
        posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')
        # print(posixvalue)
        if os.path.exists(posixvalue):
            count += 1
            existingData.append(item)
            filename = item+".json"
            print(filename, file=open("existing_files.txt", "a"))

    print(len(existingData))


    blankDict = {}
    for i in range(14,19):
        blankDict[str(i)] = []

    print(blankDict)

    for item in existingData:
        if int(item[:2]) >= 14 and item[:2] in blankDict:
            blankDict[item[:2]].append(item)

    resultDict = {}
    for i in range(14,19):
        resultDict[str(i)] = random.choices(blankDict[str(i)], k=30)


    f = open("choice.json", "w")
    json.dump(resultDict, f)
    f.close()


    for key, value in resultDict.items():
        for item in value:
            posixvalue = Path.cwd().joinpath("/home/group/cset/newAllCSjson", item)
            posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.json')
            
            with open(posixvalue) as fp:
                data = json.load(fp)
                data = convert(data)
                
                writeFile = Path.cwd().joinpath(outdir, item)
                writeFile = writeFile.with_suffix(writeFile.suffix+'.txt')

                open(writeFile, "w").write(data)


                
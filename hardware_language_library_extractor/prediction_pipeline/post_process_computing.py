import time
import re 
import string 
import spacy 
# import pandas as pd 
import numpy as np 
import math 
import copy
import os
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

import en_core_web_sm
# pd.set_option('display.max_colwidth', 200)
# import textdistance
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

BRANDNAME = ['GTX', 'Titan', 'Corsair', 'Dell', 'DFI', 'Elitegroup', 'ECS', 'EVGA', 'Foxconn', 'Gigabyte', 'MSI', 'Intel', 'Razer', 'Asus', 'Server', 'ROG', 'GeForce', 'RTX', 'Vega', 'Polaris', 'Turing', 'Alienware', 'desktop', 'laptop', 'PC', 'Pentium', 'microprocessor', 'chipset', 'Core', 'ARM', 'Atom', 'ZTE', 'Arm', 'Cortex-M', 'Iris', 'UHD', 'Celeron', 'Tiger', 'Comet', 'Ice', 'Whiskey', 'Coffee', 'Kaby', 'Acer', 'Nitro', 'Predator', 'ConceptD', 'AMD', 'Radeon', 'RX', 'Vega', 'APU', 'Ryzen', 'Opteron', 'Epyc', 'Instinct', 'Nvidia', 'CUDA', 'Tegra', 'Qualcomm', 'Tesla', 'nForce', 'GRID', 'Shield', 'linux', 'Ubuntu', 'Solaris', 'FreeBSD', 'Google', 'Brain', 'DGX', 'cluster', 'virtual', 'machine', 'cloud', 'AWS', 'Amazon', 'IBM', 'Watson', 'Fujitsu', 'Mali', 'Adreno', 'Tegra', 'Exynos', 'Snapdragon', 'Arduino', 'Apple', 'Mac', 'iMac', 'MacBook', 'Xeon', 'i5', 'i7', 'i9', 'Haswell', 'Broadwell', 'Skylake', 'Cascade', 'MediaTek', 'Helio', 'Dell', 'FPSA', 'DLP', 'Stratix']
PROCESSOR = ['FPGA', 'embedded', 'OpenGL', 'OpenCL', 'RISC', 'GPGPU']
CLOCKSPEED = ['Hz', 'GHz', 'cycle', 'clock']
VERB = ['run', 'runs', 'runtime']
PART = ['RAM', 'Processor', 'Processors', 'HDD', 'DRAM', 'SRAM', 'ROM', 'SSD', 'CPU', 'CPUS', 'GPU', 'GPUs']
TIME = ['time', 'runtime', 'FPS', 'frame', 'second', 'seconds', 'microsecond', 'microseconds', 'µs', 'millisecond','milliseconds', 'ms', 'minute','minutes', 'hour','hours','day', 'days', 'latency']


FULL_LIST = BRANDNAME + PROCESSOR + CLOCKSPEED + VERB + PART + TIME
FULL_SET_lOWER = set([word.lower() for word in FULL_LIST])
THRESHOLD = 2


def process(file):
    # basename = list(itemData.keys())[0]


    with open(file, 'r') as f:
        itemData = json.load(f)

    basename = os.path.basename(file)
    basename = basename.split(".ou")[0]



    if itemData["compute_resources"]:
        newList = []
        for sent in itemData["compute_resources"]:
            doc = nlp(sent['sentence'])
            token_set = set([tok.lower_ for tok in doc])
            match = FULL_SET_lOWER & token_set
            if len(match) >= THRESHOLD:
                sent['match'] = len(match)
                newList.append(sent)
        
        itemData["compute_resources"] = newList



    if itemData["hardware_platforms"]:
        for sent in itemData["hardware_platforms"]:
            doc = nlp(sent['sentence'])
            token_set = set([tok.lower_ for tok in doc])
            match = FULL_SET_lOWER & token_set
            if len(match) >= THRESHOLD:
                sent['match'] = len(match)
                itemData["compute_resources"].append(sent)    


    if itemData["language_libraries"]:
        for sent in itemData["language_libraries"]:
            doc = nlp(sent['sentence'])
            token_set = set([tok.lower_ for tok in doc])
            match = FULL_SET_lOWER & token_set
            if len(match) >= THRESHOLD:
                sent['match'] =len(match)
                itemData["compute_resources"].append(sent) 

            
    del itemData["language_libraries"]
    del itemData["hardware_platforms"]

    posixvalue = Path.cwd().joinpath(outpath, basename)
    posixvalue = posixvalue.with_suffix(posixvalue.suffix+'.output.json')

    with open(posixvalue, 'w') as output:
        json.dump(itemData, output, indent=4)


################### main interation###################
######################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonFilePath", default=None, help="path to the output directory") 
    parser.add_argument("--output", default=None, help="path to the output directory") 
    args = parser.parse_args()

    inpath = args.jsonFilePath
    outpath = args.output
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    selectedFiles = sorted(Path.cwd().joinpath(inpath).glob('*.output.json'))

    # with open('/home/raquib/xtest_cset/hardware_language_library_extractor/CSETreview.json', 'r') as f:
    #     selectedFiles = json.load(f)

    print("number of files to process", len(selectedFiles))
    pool = Pool()
    pool.map(process, selectedFiles)


    print("--- %s seconds ---" % (time.time() - start_time))
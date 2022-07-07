import time
import re 
import string 
import spacy 
# import pandas as pd 
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
################################
# inpath = '/home/group/cset/allCSjson'
# outpath = 'fullOutputSectionized'


def spaninformation(doc, span):
    result=[]
    tempend=span.sent.end
    result.append((span.sent.start, span.sent.end))
    
    for i in range(0,5):
        newspan=doc[tempend+1:tempend+2]
        result.append((newspan.sent.start, newspan.sent.end))
        tempend=newspan.sent.end
        
    return result


def datasetCandGen(dictFile):
    savingSectionInfo=True
    
    candidateSeedWords = [{}, {}, {}, {}, {}, {}, {}, {}]
    #this will hold the report for full pdf
    candidateSent=[] 


    for sectionKey, value in dictFile[1].items():
        doc=nlp(value)
        # for each sentence, this will also hold the dataset name seeds(refered as NN from this point)
        seedwords=['database', 'dataset', 'corpus','corpora', 'data-set', 'data-base', 'datasets', 'databases']
        matcher = Matcher(nlp.vocab) 
        for x in seedwords:
            pattern = [{'LOWER': x}] 
            matcher.add(x, None, pattern) 


        #different matcher for diff facets
        # list all seed words separated by space with both capital case and lower case(Code)
        #1
        nsubjwordString="performance paper we dataset experiment"
        nsubjwords=[token.lemma_ for token in nlp(nsubjwordString)]
        nsubj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["nsubj", "nsubjpass", "compound", "conj", "pobj"]}, 'LEMMA': {"IN": nsubjwords}
                , 'LOWER':{'NOT_IN': ["i"]}},
            {'OP': '*'},
            {'DEP': 'ROOT'}]
        nsubj_matcher.add('nsubj', None, pattern)
        #complementing matcher
        not_nsubj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["nsubj", "nsubjpass", "compound", "conj"]},'LEMMA': {"NOT_IN": nsubjwords}, 
                    'IS_DIGIT': False, 'IS_PUNCT': False, 'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False},
            {'OP': '*'},
            {'DEP': 'ROOT'}]
        not_nsubj_matcher.add('not_nsubj', None, pattern)

        #2 normal
        rootwordString="""make utilize adopt create construct include consist perfom introduce contain feed
                        is perform use implement evaluate release focus conduct train constitute"""
        rootwords=[token.lemma_ for token in nlp(rootwordString)]
        root_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"IN": rootwords}}]
        root_matcher.add('root', None, pattern) 
        #complementing matcher
        not_root_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"NOT_IN": rootwords},
                'IS_DIGIT': False, 'IS_PUNCT': False, 'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False}]
        not_root_matcher.add('not_root', None, pattern)

        #3
        objwordString="""database dataset github website repository online collection benchmark numerical study"""
        objwords=[token.lemma_ for token in nlp(objwordString)]
        obj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': {"IN": ["pobj", "dobj", "obj", "compound", "conj", "iobj"]},
                                                'LEMMA': {"IN": objwords}}]
        obj_matcher.add('obj', None, pattern)
        #complementing matcher
        not_obj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': {"IN": ["pobj", "dobj", "obj", "compound", "conj", "iobj"]},
                                                'LEMMA': {"NOT_IN": objwords}, 'IS_DIGIT': False, 'IS_PUNCT': False, 
                                                'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False}]
        not_obj_matcher.add('not_obj', None, pattern)

        #4   
        adjwordString="publicly available online large-scale constructed synthetic dataset popular constructed"
        adjwords=[token.lemma_ for token in nlp(adjwordString)]
        adj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["amod", "oprd", "advmod", "acomp"]}, 'LEMMA': {"IN": adjwords}}]
        adj_matcher.add('adj', None, pattern) 
        #complementing matcher
        not_adj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["amod", "oprd", "advmod", "acomp"]}, 'LEMMA': {"NOT_IN": adjwords}}]
        not_adj_matcher.add('not_adj', None, pattern) 

        #5
        whichwordString="""generate provide utilize adopt create construct include consist introduce contain feed
                        use release"""
        whichwords=[token.lemma_ for token in nlp(whichwordString)]
        which_matcher = Matcher(nlp.vocab)
        pattern = [{'DEP':'ROOT'}, {'OP': '*'}, {'DEP':{"IN": ["nsubj", "nsubjpass"]},'LEMMA':{"IN": ['which', 'that']}}, 
                {'OP': '*'}, {'DEP':{"IN": ["relcl", "parataxis"]}, 'LEMMA': {"IN": whichwords}}]
        which_matcher.add('which', None, pattern)
        #complementing matcher
        not_which_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP':'ROOT'}, {'OP': '*'}, {'DEP':{"IN": ["nsubj", "nsubjpass"]},'LEMMA':{"IN": ['which', 'that']}}, {'OP': '*'}, 
                {'DEP':{"IN": ["relcl", "parataxis"]}, 'LEMMA': {"NOT_IN": whichwords}}]
        not_which_matcher.add('not_which', None, pattern)

        #6 number matcher
        numberwordString="""include consist contain constitute compose comprise"""
        numberwords=[token.lemma_ for token in nlp(numberwordString)]
        number_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"IN": numberwords}}, 
                {'OP': '*'},{'DEP': 'nummod', 'LIKE_NUM':True, 'IS_ALPHA':False}]
        number_matcher.add('number', None, pattern)
        #complementing matcher
        not_number_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"NOT_IN": numberwords}}, 
                {'OP': '*'},{'DEP': 'nummod', 'LIKE_NUM':True, 'IS_ALPHA':False}]
        not_number_matcher.add('not_number', None, pattern)

        #2.2 (7) composed of type with acl tag
        aclString="compose consist comprise"
        aclwords=[token.lemma_ for token in nlp(aclString)]
        acl_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': 'acl', 'LEMMA': {"IN": aclwords}}, {'OP': '*'},{'DEP': 'prep'},
                {'OP': '*'}, {'DEP': 'nummod', 'LIKE_NUM':True}]
        acl_matcher.add('acl', None, pattern)
        #complementing matcher
        not_acl_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': 'acl', 'LEMMA': {"NOT_IN": aclwords}}, {'OP': '*'},{'DEP': 'prep'},
                {'OP': '*'}, {'DEP': 'nummod', 'LIKE_NUM':True}]
        not_acl_matcher.add('not_acl', None, pattern)

        #2.1 (8) xcomp/pcomp type, using type
        compString="evaluate using constitute provide containing"
        compwords=[token.lemma_ for token in nlp(compString)]
        comp_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP':{"IN": ["xcomp","pcomp","advcl"]}, 'LEMMA': {"IN": compwords}}]
        comp_matcher.add('comp', None, pattern)
        #complementing matcher
        not_comp_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP':{"IN": ["xcomp","pcomp","advcl"]}, 'LEMMA': {"NOT_IN": compwords}}]
        not_comp_matcher.add('not_comp', None, pattern)

        #word to exclude from NN
        exclusion = ['table', 'figure', 'sample', 'section', 'test', 'train', 'parameter', 'sql', 'cluster', 'CNN', 'RNN', 'MATLAB']

        #negative weight for these words
        third_matcher = Matcher(nlp.vocab)
        thirdNegativeWeight= ['author', 'section', 'function', 'table', 'figure', 'their', 'literature', 'work', 'related'
                            'works','literatures']
        pattern = [{'LOWER': {'IN':thirdNegativeWeight}}] 
        third_matcher.add('junkWords', None, pattern)
        pattern = [{'DEP':'neg'}]
        third_matcher.add('negativeWords', None, pattern)

        #1 is nsubj, 2 is root, 3 is obj, 4 is adj, 5 is which block, 6 is number, 7 is acl, 8 comp
    
        candidateNN=[]
        doneSpans = []

        matches = matcher(doc) #matches are each candidate sentences

        #traverse through each matches and next 5 sentence
        for match_id, start, end in matches:
            #check against doneSpans if this match is already done inside any previous match's context
            alreadyDone=False
            for doneTuple in doneSpans:
                if doneTuple[0] <= start <=doneTuple[1]:
                    alreadyDone=True

            if alreadyDone:
                continue

            span = doc[start:end]  # The matched span
            contextSpans = spaninformation(doc, span)
            doneSpans.append((contextSpans[0][0],contextSpans[-1][1]))

            # traverse through each one of the sentence in context
            for tupleSpan in contextSpans:
                #print(doc[tupleSpan[0]:tupleSpan[1]].text)

                newdoc=nlp(doc[tupleSpan[0]:tupleSpan[1]].text)
                seedmatches = matcher(newdoc)
                #how many seed matches(dataset, database this kind of words)
                if seedmatches:
                    seedSpan=newdoc[seedmatches[0][1]:seedmatches[0][2]]
                    #print("found seed", seedSpan.sent.text)

                #testing againt negative weight words, if found, skip these
                thirdmatches = third_matcher(newdoc)
                if thirdmatches:
                    #print("found negative words, ", span.sent.text)
                    continue

                #variables for this line only
                sentencePoint=0  # weight counter for each sentence, this will be saved for each 
                NNDict={}
                LinkFound=False


                #now checking against pattern matcher and creating the candidate seed words for 7 patterns
                numberOfMatches = {} # will hold number of matches for each of nsubj, root, obj, adj
                #1 is nsubj, 2 is root, 3 is obj, 4 is adj, 5 is which block, #6 is number, #2.2(7) composed of
                newmatches = nsubj_matcher(newdoc)
                numberOfMatches['nsubj']=len(newmatches)

                newmatches = root_matcher(newdoc)
                numberOfMatches['root']=len(newmatches)

                newmatches = obj_matcher(newdoc)
                numberOfMatches['obj']=len(newmatches)

                newmatches = adj_matcher(newdoc)
                numberOfMatches['adj']=len(newmatches)

                newmatches = which_matcher(newdoc)
                numberOfMatches['which']=len(newmatches)

                newmatches = number_matcher(newdoc)
                numberOfMatches['number']=len(newmatches)

                newmatches = acl_matcher(newdoc)
                numberOfMatches['acl']=len(newmatches)

                newmatches = comp_matcher(newdoc)
                numberOfMatches['comp']=len(newmatches)

                #initializing the NN, ref and links values to 0
                numberOfMatches['NN']=0
                numberOfMatches['reference']=0
                numberOfMatches['link']=0

                #checking NN, ref, links before facets checking
                #making dependency graph, use these at anything necessary
                edges = []
                for tok in doc[tupleSpan[0]:tupleSpan[1]]:
                    for child in tok.children:
                        edges.append((tok.text, child.text))
                graph = nx.Graph(edges)

                #going through the tokens one by one checking NN, ref, links before facets checking
                for tok in doc[tupleSpan[0]:tupleSpan[1]]:
                    #trying to get the NNs, seed khuje na pele NN match korbo na
                    if not tok.is_stop and not tok.is_punct \
                    and not tok.is_space and (tok.tag_ == 'NN' or tok.tag_ == 'NNP') \
                    and (tok.is_title or tok.text[0].isupper() or tok.is_upper) and len(tok.text)>2 \
                    and not any(word in str(tok.lower_) for word in exclusion) and seedmatches \
                    and tok.text != seedSpan.text and graph.has_node(tok.text) and graph.has_node(seedSpan.text):
                        #print(tok.text, "-->",tok.dep_, "-->",tok.pos_, "-->", tok.tag_, "-->", tok.ent_type_)
                        individualNNpoint = 0
                        if tok.text[-1].isdigit():
                            #this is the weight of individual NN inside each sentence
                            individualNNpoint = individualNNpoint-1

                        #check shortest dependency path here and add to the individualNNpoint
                        individualNNpoint = individualNNpoint + nx.shortest_path_length(graph, source=tok.text, target=seedSpan.text)
                        NNDict[tok.text]=individualNNpoint
                        numberOfMatches['NN'] = numberOfMatches.get('NN', 0) + 1

                    #print(tok.text, "-->",tok.dep_, "-->",tok.pos_, "-->", tok.tag_, "-->", tok.ent_type_)
                    if tok.text.startswith("#"):
                        #print(tok.text+tok.nbor().text)
                        sentencePoint+=3
                        numberOfMatches['reference'] = numberOfMatches.get('reference', 0) + 1

                    if tok.like_url: #also make sure the link can be read with space in between
                        #print("got a url"+tok.text)
                        sentencePoint+=4
                        LinkFound=True
                        numberOfMatches['link'] = numberOfMatches.get('link', 0) + 1

            
                #decide if this sentence is a valid one, if there are at least two of the values are more than 0 than it is
                #and (numberOfMatches['adj']>=1 or numberOfMatches['obj']>=1)
                #among total facets(11), at best 7 can have 0, so we need values from at least 4 facets, changed from 3
                #changed to at least 3 facets required before pwc data count(0)<=8
                validSentence=False
                if list(numberOfMatches.values()).count(0)<=9 and sum(list(numberOfMatches.values()))>=3:
                    validSentence=True

                    #populate the candidateSeedWord
                    for key, value in numberOfMatches.items():
                        if key == 'nsubj' and value >= 0:
                            newmatches = not_nsubj_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[start:start+1]
                                #print(string_id, newspan.lemma_)
                                #do some checking to get rid of the refs, single character or others here
                                candidateSeedWords[0][newspan.lemma_] = candidateSeedWords[0].get(newspan.lemma_, 0) + 1

                        elif key == 'root' and value >= 0:
                            newmatches = not_root_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[start:end]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[1][newspan.lemma_] = candidateSeedWords[1].get(newspan.lemma_, 0) + 1

                        elif key == 'obj' and value >= 0:
                            newmatches = not_obj_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[end-1:end]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[2][newspan.lemma_] = candidateSeedWords[2].get(newspan.lemma_, 0) + 1

                        elif key == 'adj' and value >= 0:
                            newmatches = not_adj_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[start:end]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[3][newspan.lemma_] = candidateSeedWords[3].get(newspan.lemma_, 0) + 1

                        elif key == 'which' and value >= 0:
                            newmatches = not_which_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[end-1:end]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[4][newspan.lemma_] = candidateSeedWords[4].get(newspan.lemma_, 0) + 1

                        elif key == 'number' and value >= 0:
                            newmatches = not_number_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[start:start+1]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[5][newspan.lemma_] = candidateSeedWords[5].get(newspan.lemma_, 0) + 1

                        elif key == 'acl' and value >= 0:
                            newmatches = not_acl_matcher(newdoc)
                            if newmatches:
                                for tok in newdoc[newmatches[0][1]:newmatches[0][2]]:
                                    if tok.dep_ == 'acl':
                                        #print(string_id, newspan.lemma_)
                                        candidateSeedWords[6][tok.lemma_] = candidateSeedWords[6].get(tok.lemma_, 0) + 1

                        elif key == 'comp' and value >= 0:
                            newmatches = not_comp_matcher(newdoc)
                            for match_id, start, end in newmatches:
                                string_id = nlp.vocab.strings[match_id]  # Get string representation
                                newspan = newdoc[end-1:end]
                                #print(string_id, newspan.lemma_)
                                candidateSeedWords[7][newspan.lemma_] = candidateSeedWords[7].get(newspan.lemma_, 0) + 1


                if validSentence:
                    #saving sentence points here
                    newdict={}
                    newdict['sentence']=doc[tupleSpan[0]:tupleSpan[1]].text
                    newdict['SentencePoint']=sentencePoint
                    newdict.update(numberOfMatches)
                    if seedmatches:
                        newdict['keyword']=seedSpan.text
                    else:
                        newdict['keyword'] = ''
                    newdict['dataset_name_candidates']=NNDict

                    #saving context sentences
                    if (tupleSpan[0]-2) >= 0 :
                        previousToken = doc[tupleSpan[0] - 2:tupleSpan[0] -1]
                        if previousToken.sent.text is not None:
                            newdict['prev_sent'] = previousToken.sent.text
                    
                    if (tupleSpan[1]+2) <= len(doc):
                        nextToken = doc[tupleSpan[1]+1:tupleSpan[1]+2]
                        if nextToken.sent.text is not None:    
                            newdict['next_sent'] = nextToken.sent.text

                    #saving the span information for website showing tupleSpan[0]:tupleSpan[1]]
                    if savingSectionInfo:
                        savingSpan=doc[tupleSpan[0]:tupleSpan[1]]
                        newdict['section_name']=sectionKey
                        newdict['start_relative_section']=savingSpan.start_char
                        newdict['end_relative_section']=savingSpan.end_char
                        newdict['start_seed']=savingSpan.start_char-savingSpan.sent.start_char
                        newdict['end_seed']=savingSpan.end_char-savingSpan.sent.start_char

                    minNNDictPoint=0
                    if NNDict:
                        minNNDictPoint = min(NNDict.values()) 
                    #res = [key for key in test_dict if test_dict[key] == temp]
                    newdict['AdjustedPoint']=sentencePoint-minNNDictPoint
                    
                        
                    #adding new computation here when in valid case, let's see noun chunk
                    if seedmatches:
                        chunkValidList={}
                        for chunk in doc[tupleSpan[0]:tupleSpan[1]].noun_chunks:
                            thisChunkValid=False
                            chunkSDPlist=[]
                            #check if this chunk has any captial case NN or worthy NN
                            for tok in chunk:
                                if not tok.is_stop and not tok.is_punct \
                                and not tok.is_space and (tok.tag_ == 'NN' or tok.tag_ == 'NNP') \
                                and (tok.is_title or tok.text[0].isupper() or tok.is_upper) and len(tok.text)>2 \
                                and not any(word in str(tok.lower_) for word in exclusion) and tok.text != seedSpan.text \
                                and graph.has_node(tok.text) and graph.has_node(seedSpan.text):
                                    thisChunkValid=True
                                    chunkSDPlist.append(nx.shortest_path_length(graph, source=tok.text, target=seedSpan.text))

                            if thisChunkValid:
                                chunkValidList[chunk.text]=min(chunkSDPlist)
                        #finally print how many chunks are there in this sentence            
                        #print(chunkValidList)
                        newdict['chunkValidList']=chunkValidList
                    
                    #inside if validSentence
                    if minNNDictPoint<5:
                        candidateSent.append(newdict)
        
    #filter out the NNs here, don't tab it, these task are at the last leverl
    countOfNN={}
    selectedNN = {}
    for sent in candidateSent:
        if sent['dataset_name_candidates']:
            minSDP=min(sent['dataset_name_candidates'].values())
            listOfSelected=list(k for k, v in sent['dataset_name_candidates'].items() if v <=(minSDP+1))
            for k in listOfSelected:
                countOfNN[k] = countOfNN.get(k, 0) + 1

    for key, value in countOfNN.items():
        if any((key in k or key.lower() in k) for k in candidateSeedWords):
            selectedNN[key]=value
        
    #selecting Noun Chunks
    chunkSelected = {}
    chunkFound = False
    for sent in candidateSent:
        if 'chunkValidList' in sent and sent['chunkValidList']:
            minSDP=min(list(sent['chunkValidList'].values()))
            for k, v in sent['chunkValidList'].items():
                if v<=3 and v == minSDP and (any(x in k for x in selectedNN.keys()) or v==1) :
                    chunkFound= True
                    chunkSelected[k]=v
                        
                
                
    #check chunk or selectedNN, if one is empty, show other, otherwise show chunks
    #filtering chunkSelected to show the final output
    #i don't think it will be very helpful
#     newchunkSelected={}
#     if chunkSelected and selectedNN:
#         for key in selectedNN.keys():
#             for k,v in chunkSelected.items():
#                 if key in k:
#                     newchunkSelected[k]=v
                    
#         for key,value in chunkSelected.items():
#             if key not in newchunkSelected:
#                 newchunkSelected[k]=v
        
    
    
    newchunkSelected=copy.deepcopy(chunkSelected)
    for key, value in chunkSelected.items():
        for k, v in chunkSelected.items():
            if key != k:
                if key in k and key in newchunkSelected:
                    #delete key
                    del newchunkSelected[key]
                    
                    
    return candidateSent, candidateSeedWords, selectedNN, chunkSelected, newchunkSelected




def sourceCandGen(dictFile):
    savingSectionInfo = True
    candidateSeedWords = [{}, {}, {}, {}]
    candidateSent=[]  #this will hold the report for each sentence, the sent, weights, keyword, weight of the NNs
    
    for sectionKey, value in dictFile[1].items():
        
        doc=nlp(value)
        # list all seed words separated by space with both capital case and lower case(Code)
        nsubjwordString="it we model implementation source code Code source-code supplementary material"
        nsubjwords=[token.lemma_ for token in nlp(nsubjwordString)]
        nsubj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["nsubj", "nsubjpass", "compound", "conj"]}, 'LEMMA': {"IN": nsubjwords}},
            {'OP': '*'},
            {'DEP': 'ROOT'}]
        nsubj_matcher.add('nsubj', None, pattern)
        #complementing matcher
        not_nsubj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["nsubj", "nsubjpass", "compound", "conj"]},'LEMMA': {"NOT_IN": nsubjwords}, 
                    'IS_DIGIT': False, 'IS_PUNCT': False, 'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False},
            {'OP': '*'},
            {'DEP': 'ROOT'}]
        not_nsubj_matcher.add('not_nsubj', None, pattern)
        
        
        rootwordString="is are find release"
        rootwords=[token.lemma_ for token in nlp(rootwordString)]
        root_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"IN": rootwords}}]
        root_matcher.add('root', None, pattern) 
        #complementing matcher
        not_root_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT', 'LEMMA': {"NOT_IN": rootwords},
                'IS_DIGIT': False, 'IS_PUNCT': False, 'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False}]
        not_root_matcher.add('not_root', None, pattern)
        

        objwordString="Github website opensource open-source implementation project page supplementary material"
        objwords=[token.lemma_ for token in nlp(objwordString)]
        obj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': {"IN": ["pobj", "dobj", "obj", "compound", "conj", "iobj"]},
                                                'LEMMA': {"IN": objwords}}]
        obj_matcher.add('obj', None, pattern)
        #complementing matcher
        not_obj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': 'ROOT'}, {'OP': '*'}, {'DEP': {"IN": ["pobj", "dobj", "obj", "compound", "conj", "iobj"]},
                                                'LEMMA': {"NOT_IN": objwords}, 'IS_DIGIT': False, 'IS_PUNCT': False, 
                                                'IS_SPACE': False, 'LIKE_NUM': False, 'LIKE_URL': False}]
        not_obj_matcher.add('not_obj', None, pattern)
            
            
        adjwordString="publicly available online opensource open-source supplementary"
        adjwords=[token.lemma_ for token in nlp(adjwordString)]
        adj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["amod", "oprd", "advmod", "acomp"]}, 'LEMMA': {"IN": adjwords}}]
        adj_matcher.add('adj', None, pattern) 
        #complementing matcher
        not_adj_matcher = Matcher(nlp.vocab) 
        pattern = [{'DEP': {"IN": ["amod", "oprd", "advmod", "acomp"]}, 'LEMMA': {"NOT_IN": adjwords}}]
        not_adj_matcher.add('not_adj', None, pattern) 
        
        
        #negative weight matching
        third_matcher = Matcher(nlp.vocab)
        thirdNegativeWeight= ['author', 'section', 'function', 'table', 'figure']
        for x in thirdNegativeWeight:
            pattern = [{'LOWER': x}] 
            third_matcher.add(x, None, pattern)
        pattern = [{'LOWER': 'pseudo'}, {"OP": "?"}, {'LOWER': 'code'}]
        third_matcher.add('pseudo-code', None, pattern)
        
        #matcher for url, footnote anf link here
        urlRefFootnote = ['#']
        matcher = Matcher(nlp.vocab) 
        for x in urlRefFootnote:
            pattern = [{'LOWER': x}] 
            matcher.add(x, None, pattern) 
        
        pattern = [{'is_currency': True}] 
        matcher.add('$', None, pattern) 
        pattern = [{'like_url': True}] 
        matcher.add('URL', None, pattern) 
        
        #1 is nsubj, 2 is root, 3 is obj, 4 is adj
        matches = matcher(doc) #matches are each candidate sentences
        processedSent=[] #this will hold the sentence's string which is already processed
        
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            #print(string_id, start, end, span.text, span.sent)
            
            #check if this sentence is already computed
            #check against candidateSent
            if str(span.sent) in processedSent:
                continue
        
            processedSent.append(str(span.sent))
            
            #testing only this sentence
            newdoc=nlp(span.sent.text)
            
            #testing againt negative weight words
            thirdmatches = third_matcher(newdoc)
            if thirdmatches:
                #print("found negative words, ", span.sent.text)
                continue
        
            #now checking against nsubj, root, obj, adj
            numberOfMatches = {} # will hold number of matches for each of nsubj, root, obj, adj
            
            newmatches = nsubj_matcher(newdoc)
            numberOfMatches['nsubj']=len(newmatches)
                
            newmatches = root_matcher(newdoc)
            numberOfMatches['root']=len(newmatches)
            
            newmatches = obj_matcher(newdoc)
            numberOfMatches['obj']=len(newmatches)
            
            newmatches = adj_matcher(newdoc)
            numberOfMatches['adj']=len(newmatches)

            #decide if this sentence is a valid one, if there are at least two of the values are more than 0 than it is
            validSentence=False
            if list(numberOfMatches.values()).count(0)<=2 and sum(list(numberOfMatches.values()))>=3 \
            and (numberOfMatches['adj']>=1 or numberOfMatches['obj']>=1):
                newdict={}
                newdict['sentence']=doc[span.sent.start:span.sent.end].text
                newdict.update(numberOfMatches)
                #***** removed candidateSent.append(newdict) from here to few lines later
                validSentence=True

                #saving context sentences
                if (span.sent.start-2) >= 0 :
                    previousToken = doc[span.sent.start - 2 : span.sent.start -1]
                    if previousToken.sent.text is not None:
                        newdict['prev_sent'] = previousToken.sent.text
                
                if (span.sent.end+2) <= len(doc):
                    nextToken = doc[span.sent.end + 1 : span.sent.end + 2]
                    if nextToken.sent.text is not None:    
                        newdict['next_sent'] = nextToken.sent.text

                if savingSectionInfo:
                    savingSpan=doc[span.sent.start:span.sent.end]
                    newdict['section_name']=sectionKey
                    newdict['start_relative_section']=savingSpan.start_char
                    newdict['end_relative_section']=savingSpan.end_char
                    newdict['start_seed']=savingSpan.start_char-savingSpan.sent.start_char
                    newdict['end_seed']=savingSpan.end_char-savingSpan.sent.start_char

                #add newdict to the candidateSent after newdict is fully formed
                #moved here from #*** place, few lines above
                candidateSent.append(newdict)

                #populate the candidateSeedWord
                for key, value in numberOfMatches.items():
                    if key == 'nsubj' and value >= 0:
                        newmatches = not_nsubj_matcher(newdoc)
                        for match_id, start, end in newmatches:
                            string_id = nlp.vocab.strings[match_id]  # Get string representation
                            newspan = newdoc[start:start+1]
                            #print(string_id, newspan.lemma_)
                            #do some checking to get rid of the refs, single character or others here
                            candidateSeedWords[0][newspan.lemma_] = candidateSeedWords[0].get(newspan.lemma_, 0) + 1

                    elif key == 'root' and value >= 0:
                        newmatches = not_root_matcher(newdoc)
                        for match_id, start, end in newmatches:
                            string_id = nlp.vocab.strings[match_id]  # Get string representation
                            newspan = newdoc[start:end]
                            #print(string_id, newspan.lemma_)
                            candidateSeedWords[1][newspan.lemma_] = candidateSeedWords[1].get(newspan.lemma_, 0) + 1

                    elif key == 'obj' and value >= 0:
                        newmatches = not_obj_matcher(newdoc)
                        for match_id, start, end in newmatches:
                            string_id = nlp.vocab.strings[match_id]  # Get string representation
                            newspan = newdoc[end-1:end]
                            #print(string_id, newspan.lemma_)
                            candidateSeedWords[2][newspan.lemma_] = candidateSeedWords[2].get(newspan.lemma_, 0) + 1

                    elif key == 'adj' and value >= 0:
                        newmatches = not_adj_matcher(newdoc)
                        for match_id, start, end in newmatches:
                            string_id = nlp.vocab.strings[match_id]  # Get string representation
                            newspan = newdoc[start:end]
                            #print(string_id, newspan.lemma_)
                            candidateSeedWords[3][newspan.lemma_] = candidateSeedWords[3].get(newspan.lemma_, 0) + 1
            
    #check candidateSent and remove the low point sentence
#     pointlist=[sent['SentencePoint'] for sent in candidateSent]
#     #print(pointlist)
#     meanvalue=np.mean(pointlist)
#     #print(meanvalue)
#     willdelete=[]
#     for ind, sent in enumerate(candidateSent):
#         if sent['SentencePoint'] < meanvalue:
#             willdelete.append(ind)
    
#     if (len(candidateSent)-len(willdelete))>1:
#         candidateSent=[e for i, e in enumerate(candidateSent) if i not in willdelete]
    
    return candidateSent, candidateSeedWords


##############################################################################################
##############################################################################################



# dataAllSentenceByFile, dataTotalCandSeeds, dataDenominator, NNResult
def process(file):
    with open(file, 'r') as f:
        mainlist = json.load(f)
        
    if DataActive:
        datasetCandSentences, datacandSeedWords, selectedNN, chunkSelected, \
        consolidatedchunkSelected =datasetCandGen(mainlist)
        
        dataAllSentenceByFile[mainlist[0]['basename']] = datasetCandSentences
        NNResult[mainlist[0]['basename']] = consolidatedchunkSelected
        dataDenominator.value=dataDenominator.value+len(datasetCandSentences)

        # print(datacandSeedWords)
        #saving candidate seed words to totalCandSeeds
        for i in range(0,8):
            for key, value in datacandSeedWords[i].items():
                # print(dataTotalCandSeeds[i].get(key, 0))
                dataTotalCandSeeds[i][key] = dataTotalCandSeeds[i].get(key, 0) + value

        # print(dataTotalCandSeeds)
    
    if SourceActive:
        #saving allSentenceByFile
        sourceCode, candSeed=sourceCandGen(mainlist)
        
        sourceAllSentenceByFile[mainlist[0]['basename']] = sourceCode
        sourceDenominator.value=sourceDenominator.value+len(sourceCode)

        #saving candidate seed words to totalCandSeeds
        for i in range(0,4):
            for key, value in candSeed[i].items():
                sourceTotalCandSeeds[i][key] = sourceTotalCandSeeds[i].get(key, 0) + value

        


################### main interation###################
######################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonFilePath", default="/home/raquib/SLIE/data/JSONs/", help="path to the input directory") 
    parser.add_argument("--output", default="/home/raquib/SLIE/output/facetBased", help="path to the output directory") 
    parser.add_argument("--SourceInActive", action='store_false', help="indicate if Source Code Extraction is inactive") 
    parser.add_argument("--SourceLearning", default=False, help="indicate if Source Code Learning phase is active, True of False (Currently Disabled)")
    parser.add_argument("--DataInActive", action='store_false', help="indicate if Dataset Extraction is inactive") 
    parser.add_argument("--DataLearning", default=False, help="indicate if Dataset Learning phase is active, True of False (Currently Disabled)")
    parser.add_argument("--n_core", default=80, help="number of CPU cores") 
    parser.add_argument("--n_iteration", default=1, help="number of itearations in Learning Phase (Currently Disabled)") 
    args = parser.parse_args()

    
    inpath = args.jsonFilePath
    outpath = args.output
    
    numberOfCore = int(args.n_core)
    numberOfIteration = args.n_iteration

    SourceActive = args.SourceInActive
    SourceLearning = args.SourceLearning
    DataActive = args.DataInActive
    DataLearning = args.DataLearning

    print("Facet-based extraction with the following parameters")
    print("input:", inpath)
    print("output:", outpath)
    print("Number of Core using", numberOfCore, "SourceActive", SourceActive, "DataActive", DataActive)
    

    for iteration in range(0,numberOfIteration):
        manager = multiprocessing.Manager()

        if SourceActive:
            sourceTotalCandSeeds = manager.list([manager.dict() for _ in range(4)])
            sourceAllSentenceByFile = manager.dict()
            sourceDenominator = Value('i', 0)

        
        #dataset candidate variables
        if DataActive:
            dataTotalCandSeeds = manager.list([manager.dict() for _ in range(8)])
            dataAllSentenceByFile = manager.dict()
            NNResult = manager.dict()
            dataDenominator=Value('i', 0)


        selectedFiles = sorted(Path.cwd().joinpath(inpath).glob('*.json'))

        print("number of files to process", len(selectedFiles))
        pool = Pool(numberOfCore)
        pool.map(process, selectedFiles)
        
        
        #printing source variables
        if SourceActive:
            # tempSourceSeedWord = []
            # for i in range(4):
            #     tempSourceSeedWord.append({k: v for k, v in sorted(sourceTotalCandSeeds[i]._getvalue().items(), key=lambda item: item[1], reverse=True)})
            # with open(Path.cwd().joinpath(outpath, 'SortedSourceSeedWords.json'), 'w') as output:
            #     json.dump(tempSourceSeedWord, output)

            # with open(Path.cwd().joinpath(outpath,'sourceSeedWords.json'), 'w') as output:
            #     json.dump([sourceTotalCandSeeds[i]._getvalue() for i in range(4)], output, indent=4)

            with open(Path.cwd().joinpath(outpath,'sourceSentences.json'), 'w') as output:
                json.dump(sourceAllSentenceByFile._getvalue(), output, indent=4, sort_keys=True)
            
            print("source sentence count", sourceDenominator.value)
        
        if SourceLearning:
            for i in range(0,4):
                for key, value in sourceTotalCandSeeds[i].items():
                    score = value/sourceDenominator
                    print(score)
                    if score>0.5 :
                        sourceListOfSeedWords[i]=sourceListOfSeedWords[i]+str(key)

        
        #printing data variables
        if DataActive:
            # tempSeedWord = []
            # for i in range(8):
            #     tempSeedWord.append({k: v for k, v in sorted(dataTotalCandSeeds[i]._getvalue().items(), key=lambda item: item[1], reverse=True)})
            # with open(Path.cwd().joinpath(outpath, 'SortedDataSeedWords.json'), 'w') as output:
            #     json.dump(tempSeedWord, output)
            
            
            # with open(Path.cwd().joinpath(outpath,'dataSeedWords.json'), 'w') as output:
            #     json.dump([dataTotalCandSeeds[i]._getvalue() for i in range(8)], output, indent=4)
            # # for i in range(0,8):
            # #     print(dataTotalCandSeeds[i])

            with open(Path.cwd().joinpath(outpath,'dataSentences.json'), 'w') as output:
                json.dump(dataAllSentenceByFile._getvalue(), output, indent=4, sort_keys=True)
            # print(dataAllSentenceByFile)
            
            print("data sentence count", dataDenominator.value)
            with open(Path.cwd().joinpath(outpath,'dataNNResult.json'), 'w') as output:
                json.dump(NNResult._getvalue(), output, indent=4, sort_keys=True)
            # print(NNResult)
        
        if DataLearning:
            for i in range(0,8):
                for key, value in dataTotalCandSeeds[i].items():
                    score = value/sourceDenominator
                    print(score)
                    if score>0.5 :
                        dataListOfSeedWords[i]=dataListOfSeedWords[i]+str(key)

    print("--- %s seconds ---" % (time.time() - start_time))
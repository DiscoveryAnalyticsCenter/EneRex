import json
from pathlib import Path
import time
import argparse

from fuzzywuzzy import fuzz
import networkx as nx

start_time = time.time()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputFile", default=None, help="path to the input file") 
    parser.add_argument("--output", default=None, help="path to the output directory") 
    
    args = parser.parse_args()
    inputFile = args.inputFile
    outpath = args.output

    data = []
    with open(inputFile, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))


    result = {}
    resultSents = {}

    hardConstraint = 0
    softConstraint = 0
    oneBlank = 0


    for item in data:
        
        totalSents = []
        for sent in item['sentences']:
            totalSents.extend(sent)

        selectedTask = []
        selectedMethod = []
        selectedSents = []

        for i, sent in enumerate(item['predicted_relations']):
            found = False

            NERDict = {}

            for ner in item['predicted_ner'][i]:
                NERDict[(ner[0], ner[1])] = ner
            # print(NERDict)

            for relation in sent:
                if relation[4] == "USED-FOR":
                    A = (relation[0], relation[1])
                    B = (relation[2], relation[3])

                    selectedNER1 = None
                    selectedNER2 = None

                    if A in NERDict:
                        selectedNER1 = NERDict[A]
                    if B in NERDict:
                        selectedNER2 = NERDict[B]

                    
                    if selectedNER1 and selectedNER2:

                        #CHECK if satisfy hard limit
                        if (selectedNER1[2] == 'Task' and selectedNER2[2] == 'Method'):
                            selectedTask.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                            selectedMethod.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                            found = True
                            hardConstraint += 1

                        elif (selectedNER2[2] == 'Task' and selectedNER1[2] == 'Method'):
                            selectedTask.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                            selectedMethod.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                            found = True
                            hardConstraint += 1

                        # Soft Limit
                        #when both arguments are NERs but one of them is not Task/Method 
                        elif (selectedNER1[2] in ['Task', 'Method'] and selectedNER2[2] not in ['Task', 'Method']):
                            
                            #work with selectedNER2[2]
                            for x in sent:
                                if x[4] in ["PART-OF", "FEATURE-OF", "HYPONYM-OF"] \
                                and ( (x[0], x[1]) == (selectedNER2[0], selectedNER2[1]) \
                                or (x[2], x[3]) == (selectedNER2[0], selectedNER2[1]) ) :
                                    
                                    if (x[0], x[1]) == (selectedNER2[0], selectedNER2[1]) and (x[2], x[3]) in NERDict:
                                        #search for (x[2], x[3])
                                        nervalue = NERDict[(x[2], x[3])][2]
                                        if nervalue == 'Task' and nervalue != selectedNER1[2]:
                                            selectedTask.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            selectedMethod.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                                            found = True
                                            softConstraint += 1

                                        elif nervalue == 'Method' and nervalue != selectedNER1[2]:
                                            selectedTask.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            found = True
                                            softConstraint += 1

                                    elif (x[2], x[3]) == (selectedNER2[0], selectedNER2[1]) and (x[0], x[1]) in NERDict:
                                        #search for (x[0], x[1])
                                        nervalue = NERDict[(x[0], x[1])][2]
                                        if nervalue == 'Task' and nervalue != selectedNER1[2]:
                                            selectedTask.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                                            found = True
                                            softConstraint += 1

                                        elif nervalue == 'Method' and nervalue != selectedNER1[2]:
                                            selectedTask.append(" ".join(totalSents[selectedNER1[0]:selectedNER1[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            found = True
                                            softConstraint += 1

                        elif (selectedNER1[2] not in ['Task', 'Method'] and selectedNER2[2] in ['Task', 'Method']):
                            
                            #work with selectedNER1[2]
                            for x in sent:
                                if x[4] in ["PART-OF", "FEATURE-OF", "HYPONYM-OF"] \
                                and ( (x[0], x[1]) == (selectedNER1[0], selectedNER1[1]) \
                                or (x[2], x[3]) == (selectedNER1[0], selectedNER1[1]) ) :
                                    
                                    if (x[0], x[1]) == (selectedNER1[0], selectedNER1[1]) and (x[2], x[3]) in NERDict:
                                        #search for (x[2], x[3])
                                        nervalue = NERDict[(x[2], x[3])][2]
                                        if nervalue == 'Task' and nervalue != selectedNER2[2]:
                                            selectedTask.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            selectedMethod.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                                            found = True
                                            softConstraint += 1

                                        elif nervalue == 'Method' and nervalue != selectedNER2[2]:
                                            selectedTask.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            found = True
                                            softConstraint += 1

                                    elif (x[2], x[3]) == (selectedNER1[0], selectedNER1[1]) and (x[0], x[1]) in NERDict:
                                        #search for (x[0], x[1])
                                        nervalue = NERDict[(x[0], x[1])][2]
                                        if nervalue == 'Task' and nervalue != selectedNER2[2]:
                                            selectedTask.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                                            found = True
                                            softConstraint += 1

                                        elif nervalue == 'Method' and nervalue != selectedNER2[2]:
                                            selectedTask.append(" ".join(totalSents[selectedNER2[0]:selectedNER2[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            found = True
                                            softConstraint += 1
                    

                    #if one of the selectedNER is blank but other is task/method
                    elif selectedNER1 or selectedNER2:
                        
                        presentSpan = None
                        blankSpan = None
                        if selectedNER1:
                            presentSpan = selectedNER1
                            blankSpan = B 
                        elif selectedNER2:
                            presentSpan = selectedNER2
                            blankSpan = A
                        
                        if presentSpan[2] in ['Task', 'Method']:
                            for x in sent:
                                if x[4] in ["PART-OF", "FEATURE-OF", "HYPONYM-OF"] \
                                and ( (x[0], x[1]) == blankSpan \
                                or (x[2], x[3]) == blankSpan ) :
                                    
                                    if (x[0], x[1]) == blankSpan and (x[2], x[3]) in NERDict:
                                        #search for (x[2], x[3])
                                        nervalue = NERDict[(x[2], x[3])][2]
                                        if nervalue == 'Task' and nervalue != presentSpan[2]:
                                            selectedTask.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            selectedMethod.append(" ".join(totalSents[presentSpan[0]:presentSpan[1]+1]))
                                            found = True
                                            oneBlank += 1

                                        elif nervalue == 'Method' and nervalue != presentSpan[2]:
                                            selectedTask.append(" ".join(totalSents[presentSpan[0]:presentSpan[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[2]:x[3]+1]))
                                            found = True
                                            oneBlank += 1

                                    elif (x[2], x[3]) == blankSpan and (x[0], x[1]) in NERDict:
                                        #search for (x[0], x[1])
                                        nervalue = NERDict[(x[0], x[1])][2]
                                        if nervalue == 'Task' and nervalue != presentSpan[2]:
                                            selectedTask.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[presentSpan[0]:presentSpan[1]+1]))
                                            found = True
                                            oneBlank += 1

                                        elif nervalue == 'Method' and nervalue != presentSpan[2]:
                                            selectedTask.append(" ".join(totalSents[presentSpan[0]:presentSpan[1]+1]))
                                            selectedMethod.append(" ".join(totalSents[x[0]:x[1]+1]))
                                            found = True
                                            oneBlank += 1

                    

            if found:
                selectedSents.append(" ".join(item['sentences'][i]))

                # check if this sentence is hard/soft/oneblank
                
                
                
                


        
        # print("tasks:", len(set(selectedTask)), set(selectedTask))
        # print("methods:", len(set(selectedMethod)), set(selectedMethod))
        # print()

        #saving result to result dictionary to print
        tempDict = {}
        tempDict['task_application'] = list(set(selectedTask))
        tempDict['method'] = list(set(selectedMethod))

        result[item['doc_key']] = tempDict
        resultSents[item['doc_key']] = selectedSents


    newResult = {}
    # group them here
    for key, value in result.items():
        # task_app
        tempDict = {}
        taskList = value["task_application"]
        G = nx.Graph()
        G.add_nodes_from(taskList)
        for x in taskList:
            for y in taskList:
                if x == y:
                    continue
                # print(x,y,fuzz.WRatio(x,y),fuzz.UWRatio(x,y),fuzz.ratio(x,y),fuzz.partial_ratio(x,y),fuzz.token_set_ratio(x,y))
                if fuzz.UWRatio(x,y) > 85:
                    G.add_edge(x, y)

        comps = [list(comp) for comp in nx.connected_components(G)]
        tempDict["task_application"] = comps
        

        # method
        methodList = value["method"]
        G = nx.Graph()
        G.add_nodes_from(methodList)
        for x in methodList:
            for y in methodList:
                if x == y:
                    continue
                # print(x,y,fuzz.WRatio(x,y),fuzz.UWRatio(x,y),fuzz.ratio(x,y),fuzz.partial_ratio(x,y),fuzz.token_set_ratio(x,y))
                if fuzz.UWRatio(x,y) > 85:
                    G.add_edge(x, y)

        comps = [list(comp) for comp in nx.connected_components(G)]
        tempDict["method"] = comps
        newResult[key] = tempDict
        


    #writing the final result values by arxiv ids
    with open(Path.cwd().joinpath(outpath, "task_app_method.json"), 'w') as output:
        json.dump(newResult, output, indent=4, sort_keys=True)

    with open(Path.cwd().joinpath(outpath, "sentences_task_app_method.json"), 'w') as output:
        json.dump(resultSents, output, indent=4, sort_keys=True)  

        
    print(hardConstraint, softConstraint, oneBlank)


    print("--- %s seconds ---" % (time.time() - start_time))
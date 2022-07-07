import time
import os
import json
import glob
from pathlib import Path
from os.path import isfile, join
import argparse
start_time = time.time()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonFilesPath", default='data/JSONs', help="path to the input JSON files")  
    parser.add_argument("--feature12Path", default='feature12/output/transformer', help="path to output of feature12")  
    parser.add_argument("--feature345Path", default='feture345/outputData', help="path to output of feature345")  
    args = parser.parse_args()

    jsonFilesPath = args.jsonFilesPath
    feature12Path = args.feature12Path
    feature345Path = args.feature345Path

    #All JSON files
    files = sorted(Path.cwd().joinpath(jsonFilesPath).glob('*.json'))



    with open(Path.cwd().joinpath(feature12Path, 'dataSentences.json'), 'r') as f:
        datasetSentences = json.load(f)

    with open(Path.cwd().joinpath(feature12Path, "dataNNResult.json"), 'r') as f:
        datasetNames = json.load(f)

    # with open(Path.cwd().joinpath(feature12Path, "dataFilteredLink.json"), 'r') as f:
    #     datasetURLs = json.load(f)

    with open(Path.cwd().joinpath(feature12Path, "sourceSentences.json"), 'r') as f:
        sourceCodeSentences = json.load(f)

    with open(Path.cwd().joinpath(feature12Path, "sourceFilteredLink.json"), 'r') as f:
        sourceCodeURLs = json.load(f)

    with open(Path.cwd().joinpath(feature345Path, "task_app_method.json"), 'r') as f:
        tasksMethods = json.load(f)

    with open(Path.cwd().joinpath(feature345Path, "sentences_task_app_method.json"), 'r') as f:
        sentTasksMethods = json.load(f)

    # with open('/home/group/cset/extracted_sentences/output/hardwareSentences.json', 'r') as f:
    #     hardware = json.load(f)

    # with open(Path.cwd().joinpath(inpath, 'll.json'), 'r') as f:
    #     languagelibrary = json.load(f)

    # with open(Path.cwd().joinpath(inpath, 'cr.json'), 'r') as f:
    #     computedResource = json.load(f)

    finalReport = {}
    
    for item in files:
        base_name = os.path.basename(item)
        base_name = base_name[0:-5]

        with open(item, 'r') as f:
            mainlist = json.load(f)

        report = {}
        report.update(mainlist[0])

        if base_name in datasetSentences:
            report['dataset_Sentences'] = datasetSentences[base_name]

        if base_name in datasetNames:
            report['dataset_Names'] = datasetNames[base_name]

        # if base_name in datasetURLs:
        #     report['dataset_URLs'] = datasetURLs[base_name]


        if base_name in sourceCodeSentences:
            report['sourceCode_Sentences'] = sourceCodeSentences[base_name]

        if base_name in sourceCodeURLs:
            report['sourceCode_URLs'] = sourceCodeURLs[base_name]

        if base_name in tasksMethods:
            report['app_area'] = tasksMethods[base_name]['App_area']

        if base_name in tasksMethods:
            report['task_application'] = tasksMethods[base_name]['task_application']

        if base_name in tasksMethods:
            report['method'] = tasksMethods[base_name]['method']

        if base_name in sentTasksMethods:
            report['task_method_sentences'] = sentTasksMethods[base_name]

        # if base_name in hardware:
        #     report['hardware'] = hardware[base_name]

        finalReport[base_name] = report

    

    with open(Path.cwd().joinpath('final_output.json'), 'w') as output:
        json.dump(finalReport, output, indent=4)

    print("--- %s seconds ---" % (time.time() - start_time))

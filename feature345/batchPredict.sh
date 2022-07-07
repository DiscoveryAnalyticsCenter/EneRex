# # making txt files from input JSON files. By default these are included: title, abstract, introduction, conclusion, discussion, remarks
# # use --onlyAbstract to use only title and abstract
# # use --includeAllSe to include all sections
# python preparedata.py \
# --input ../data/JSONs \
# --output txtForDygie \
# --n_core 40


# # format data to convert these txt files into a single jsonl file $largeInputFile for prediction
# # passing --cleanDir will empty the txtForDygie/ after creating the jsonl file
# export largeInputFile="outputBatchMode/processedInput.jsonl"

# python scripts/new-dataset/format_new_dataset_own.py \
# txtForDygie \
# $largeInputFile \
# --cleanDir



# # batchPreprocessing to create a batch of jsonl files that is easier to process in predict loop
# # use --preprocess to indicate its preprocess mode. converts a large jsonl file to batch of smaller jsonl files 
# # Same script handles both preprocess and post process
# # directory holding batches of smaller jsonl files
# export batchDataDir="outputBatchMode/batchData/"

# python batchPreProcess.py \
# --jsonlFile $largeInputFile \
# --batchDir $batchDataDir \
# --preprocess \
# --batchSize 500


# # prediction loop
# # batchoutputDir directory holds batches of ouptut jsonl files from prediction loop
# # make sure the last slash is there cause in the next loop we just append filename to this path
# export batchoutputDir="outputBatchMode/batchOutput/"

# FILES=outputBatchMode/batchData/*
# for inputFile in $FILES
# do
    
#     var=`basename $inputFile`
#     outputFile="${batchoutputDir}${var}"

#     allennlp predict pretrained/scierc-lightweight.tar.gz \
#     $inputFile \
#     --predictor dygie \
#     --include-package dygie \
#     --use-dataset-reader \
#     --output-file $outputFile \
#     --cuda-device 0 \
#     --silent
 
# done


# #batchPreprocessing to merge and create single output jsonl file from a batch of output jsonl files
# python batchPreProcess.py \
# --batchOutputDir $batchoutputDir \
# --mergedResultFile outputBatchMode/mergedBatchResult.jsonl


# #our filter on the dygiepp result
# #creates task_app_method.json and sentences_task_app_method.json in outputBatchMode/ directory
# python postProcess_final.py \
# --inputFile outputBatchMode/mergedBatchResult.jsonl \
# --output outputBatchMode


#assign application area to results
CUDA_VISIBLE_DEVICES=0 python application_area.py \
--taxonomyDir taxonomySeeds \
--input outputBatchMode
# making txt files from input JSON files. By default these are included: title, abstract, introduction, conclusion, discussion, remarks
# use --onlyAbstract to use only title and abstract
# use --includeAllSec to include all sections
python preparedata.py \
--input ../data/JSONs \
--output txtForDygie \
--n_core 40


# these are input and output file name for prediction
export inputFile="outputSingleMode/processedInput.jsonl"
export outputFile="outputSingleMode/DygieppOutput.jsonl"

# Combine the txt files inside txtForDygie/ into a single jsonl file for prediction
# passing --cleanDir will empty the txtForDygie/ after creating the jsonl file
python scripts/new-dataset/format_new_dataset_own.py \
txtForDygie \
$inputFile \
--cleanDir


# prediction 
allennlp predict pretrained/scierc-lightweight.tar.gz \
$inputFile \
--predictor dygie \
--include-package dygie \
--use-dataset-reader \
--output-file $outputFile \
--cuda-device 0 \
--silent


# our filter on the dygiepp result
# creates sentences_task_app_method.json and sentences_task_app_method.json in outputSingleMode directory
python postProcess_final.py \
--inputFile $outputFile \
--output outputSingleMode


# assign application area to results in outputSingleMode/resultTaskMethod.json
# python application_area.py \
# --taxonomyDir taxonomySeeds \
# --input outputSingleMode

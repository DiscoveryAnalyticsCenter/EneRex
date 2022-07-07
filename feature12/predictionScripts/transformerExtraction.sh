#!/bin/bash

# predict and save source code related sentences & dataset related sentences
# predicting sentences by classifier, creates dataSentences.json and sourceSentences.json files
# Please update modelLocation.json file with corresponding directory names for model and tempDataDir
CUDA_VISIBLE_DEVICES=0 python transformer_predictClassifier_nonsection.py \
--jsonFilePath ../data/JSONs/ \
--output output/transformer/ 


# NER model running for dataset feature. Please follow predict_NERScript.sh if you want to tweak other options
# it will output named entities, save spans in the dataSentence file, 
# also creates dataNNResult.json which contains all compiled result of Named entities for each file
bash NERmodel/predict_NERScript.sh


# # Findings Links for Source codes, this will create sourceFilteredLink.json file in the output folder
python findingLinks_source.py \
--input output/transformer/ \
--output output/transformer/ \
--n_core 40 \
--jsonFilePath ../data/JSONs/


# # Findings Links/Refered paper for dataset names, this will create dataFilteredLink.json file in the output folder
# python findingLinks_data.py \
# --input output/transformer/ \
# --output output/transformer/ \
# --n_core 40 \
# --jsonFilePath ../data/JSONs/







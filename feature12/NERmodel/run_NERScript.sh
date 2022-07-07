#!/bin/bash

export MAX_LENGTH=256
export BERT_MODEL=allenai/scibert_scivocab_uncased
export DATA_DIR=/home/raquib/SLIE/NERmodel/data/
export LABEL_DIR=/home/raquib/SLIE/NERmodel/data/labels.txt
export OUTPUT_DIR=/home/raquib/SLIE/NERmodel/OutputData_256
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

# prepare data into train.txt.tmp and dev.txt.tmp by this script
# dataSentences.json with full data not given in github but the training data are already created
# in the data directory inside NERmodel so no need to run this python file again

# python predictNER_PrepareData.py \
# --sentenceFile /home/raquib/Extraction/fullOutputSectionizedContext/dataSentences.json \
# --datasetNameFile_FacetBased /home/raquib/Extraction/fullOutputSectionizedContext/dataNNResult.json \
# --datasetNameFile_Given \
# --trainMode


#Run some default preprocessing to convert train.txt.tmp to data.txt
#bert model is the allenai scibert one
CUDA_VISIBLE_DEVICES=1 python3 run_preprocess.py data/train.txt.tmp $BERT_MODEL $MAX_LENGTH > data/train.txt
CUDA_VISIBLE_DEVICES=1 python3 run_preprocess.py data/dev.txt.tmp $BERT_MODEL $MAX_LENGTH > data/dev.txt

#create label.txt from data and dev
cat data/train.txt data/dev.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > data/labels.txt


echo Data Preprocessing Finished, Starting Training

CUDA_VISIBLE_DEVICES=1 python3 run_ner.py \
--data_dir $DATA_DIR \
--labels $LABEL_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--overwrite_output_dir

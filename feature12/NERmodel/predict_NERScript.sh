#!/bin/bash

# This script is used for only prediction in NER model
# See run_NERScript.sh if you want to train the model again

export MAX_LENGTH=128
export BERT_MODEL=NERmodel/OutputData_128/
export DATA_DIR=NERmodel/data/
export LABEL_DIR=NERmodel/data/labels.txt
export OUTPUT_DIR=NERmodel/predictions/
export BATCH_SIZE=16
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=1

#it will read output of sentence classifier and make temporary test data in NER model data dir
python NERmodel/predictNER_PrepareData.py \
--sentenceFile output/transformer/dataSentences.json \
--outputDir $DATA_DIR


#actual prediction script
CUDA_VISIBLE_DEVICES=0 python3 NERmodel/predict_ner.py \
--data_dir $DATA_DIR \
--labels $LABEL_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_predict \
--overwrite_output_dir \
--overwrite_cache

# it will read output of prediction and save spans in the dataSentence file, 
# also creates dataNNResult.json which contains all compiled result of Named entities for each file
python NERmodel/predictNER_updateSentenceFile.py \
--sentenceFile output/transformer/dataSentences.json \
--outputDir output/transformer/ \
--inputDir $OUTPUT_DIR
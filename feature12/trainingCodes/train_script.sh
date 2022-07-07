export DATA_DIR=/home/raquib/SLIE/classifierTransformer/noncontext/datasetData
export OUTPUT_DIR=/home/raquib/SLIE/classifierTransformer/noncontext/datasetOutput
export TASK_NAME=cola

CUDA_VISIBLE_DEVICES=0 python trainCodeNew.py \
  --model_name_or_path allenai/scibert_scivocab_uncased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --max_seq_length 256 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --output_dir $OUTPUT_DIR



  # python -m torch.distributed.launch \
  #   --nproc_per_node 4 trainCodeNew.py \
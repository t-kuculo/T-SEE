#!/bin/bash

#SBATCH --job-name=relation_extraction         # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --gpus=1                        # Number of GPUs to use
#SBATCH --output=training_%j.log     # Standard output and error log (%j expands to jobId)


# Run the script
python transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --ddp_timeout 18000 \
  --do_eval \
  --train_file training_data/wde_sparse_re_train2.json \
  --validation_file  training_data/wde_sparse_re_dev2.json \
  --test_file  training_data/wde_sparse_re_test2.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 30 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir models/wde_re_model \
  --save_steps 5000 \
  --version_2_with_negative

#!/bin/bash -f 

OUTPUT_DIR="/data_root/outputs/local_gpu_balanced_BertFc_seq_64_lr_1em5_i1epoch/"

python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir $OUTPUT_DIR \
                 --max_seq_length 64 \
                 --do_train \
		 --train_file "balanced_train.csv" \
                 --train_batch_size 32 \
                 --eval_batch_size 32 \
                 --num_train_epochs 1 \
                 --gradient_accumulation_steps=10 \
		 --learning_rate 1e-5 \
                 --fp16

cp run_classifier.py $OUTPUT_DIR

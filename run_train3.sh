#!/bin/bash -f 

OUTPUT_DIR="/data_root/outputs/local_gpu_balanced_BertFCWithExcl2_maxseq-128_lr_2em5_1epoch/"

python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir $OUTPUT_DIR \
                 --max_seq_length 128 \
                 --do_train \
		 --train_file "comment_exclamation.csv" \
                 --train_batch_size 32 \
                 --eval_batch_size 32 \
                 --num_train_epochs 1 \
                 --gradient_accumulation_steps=6 \
		 --learning_rate 2e-5 \
                 --fp16

cp run_classifier.py $OUTPUT_DIR

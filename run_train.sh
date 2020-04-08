#!/bin/bash -f 


python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir "/data_root/outputs/2gpu_balanced_10epoch/"\
                 --max_seq_length 512 \
                 --do_train \
		 --train_file "balanced_train.csv" \
                 --train_batch_size 320 \
                 --eval_batch_size 128 \
                 --num_train_epochs 10 \
                 --gradient_accumulation_steps=20 \
                 --fp16

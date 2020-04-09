#!/bin/bash -f 


python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir "/data_root/outputs/2gpu_large_tr-batch_64_ev-batch_8_lr-1e-6_5epoch/"\
                 --max_seq_length 128 \
                 --do_train \
		 --train_file "large_train.csv" \
                 --train_batch_size 128 \
                 --eval_batch_size 32 \
                 --num_train_epochs 5 \
                 --gradient_accumulation_steps=10 \
		 --learning_rate 1e-6 \
                 --fp16

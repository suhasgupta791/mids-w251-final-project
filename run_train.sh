#!/bin/bash -f 

python run_classifier.py --data_dir "data/nlp.cs.princeton.edu/SARC/2.0/files/" \
		  --bert_model "bert-base-uncased" \
                         --bert_model_path "/root/data/BERT/uncased_L-12_H-768_A-12-pytorch" \
			 --output_dir "/root/data/BERT/output_dir" \
                         --max_seq_length 128 \
                         --do_train \
                         --train_batch_size 72 \
                         --eval_batch_size 32 \
                         --num_train_epochs 3 \
                         --gradient_accumulation_steps 4 \
                         --fp16


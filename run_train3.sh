#!/bin/bash -f 

OUTPUT_DIR="/data_root/outputs/v100_fulldata_BertFCWithExcl2_MSE_loss_dp-0p5/"
#OUTPUT_DIR="/data_root/outputs/test"

cp run_classifier.py $OUTPUT_DIR
cp run_train3.sh $OUTPUT_DIR
python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir $OUTPUT_DIR \
                 --max_seq_length 128 \
                 --do_train \
		 --train_file "comment_exclamation.csv" \
                 --train_batch_size 148 \
                 --eval_batch_size 128 \
                 --num_train_epochs 1 \
                 --gradient_accumulation_steps=15 \
		 --cross_validation_interval=40 \
		 --checkpoint_interval=45 \
		 --learning_rate 2e-6 \
		 --num_warmup_steps 15 \
                 --fp16

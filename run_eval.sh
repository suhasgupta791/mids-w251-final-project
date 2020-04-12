#!/bin/bash -f 

OUTPUT_DIR="/data_root/outputs/test_predict/"

python run_classifier.py \
		 --data_dir "/data_root/nlp.cs.princeton.edu/SARC/2.0/files/" \
		 --bert_model "bert-base-uncased" \
                 --bert_model_path "/data_root/BERT/uncased_L-12_H-768_A-12-pytorch/" \
		 --output_dir $OUTPUT_DIR \
                 --max_seq_length 128 \
                 --do_eval \
		 --trained_model_path "/data_root/BERT/v100_full-data_BertFCWithExcl2_maxseq-128_lr_2em5_1epoch/trained_model.pt"\
		 --test_file "pol_data.csv" \
                 --eval_batch_size 8 \
                 --fp16

cp run_classifier.py $OUTPUT_DIR
cp run_train3.sh $OUTPUT_DIR

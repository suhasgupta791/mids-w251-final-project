#/bin/bash -f

BERT_BASE_DIR="$HOME/w251/mids-w251-final-project/data/BERT/uncased_L-12_H-768_A-12"

transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

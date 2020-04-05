# coding: utf-8

import itertools
import random
import numpy as np
import sys, os
import pandas as pd 
import torch
from torchsummary import summary
from torchtext import data
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook
from IPython.core.interactiveshell import InteractiveShell
import warnings
warnings.filterwarnings(action='once')
import pickle
import shutil
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import logger

# Import transformers specific packages
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import  BertForSequenceClassification, BertForTokenClassification
from transformers import AdamW,get_linear_schedule_with_warmup, pipeline

# Import Models
from models.Transformers.BertModel import *

# Import utils
from utils.utils import *

# Check if cuda is available
# Set the device and empty cache
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device =='cuda':
    from apex import amp
    torch.cuda.empty_cache()

print(device)

def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--bert_model_path", default=None, type=str, required=False,
                        help="Directory containing the converterd pytorch bert model on disk.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    

    seed = random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    logdir = args.output_dir+'/logs'

    # Bert Model 
    bert_model_name = args.bert_model

    max_seq_len = args.max_seq_length

    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None

    # Convert TF checkpoint to pytorch checkpoint and then use as input to class object
    if args.bert_model_path:
        bert_model_path = args.bert_model_path

    # Load data
    data_path = args.data_dir
    #train_file_name = data_path+'small_train.csv'
    #test_file_name  = data_path+'small_test.csv'
    train_file_name = data_path+'balanced_train.csv'
    # test_file_name  = data_path+'balanced_test.csv'
    all_train_df = pd.read_csv(train_file_name)
    #test_df = pd.read_csv(test_file_name)

    
    # Create a train, valid split
    train_data,train_labels,valid_data,valid_labels = createTestTrainSplit(all_train_df, test_size=0.2,seed=seed)

    train_size,valid_size= len(train_data),len(valid_data)
    print(train_size,valid_size)

    # Create a model object
    bert_model1=Bert_Model(train_df=train_data,
                          bert_model_name=bert_model_name,
                          bert_model_path=bert_model_path,
                          tokenizer=tokenizer,
                          max_seq_length=max_seq_len)

    print("--Tokenizing--")
    train_data_tokenized,train_attention_mask = tokenize(tokenizer,train_data)
    valid_data_tokenized,valid_attention_mask = tokenize(tokenizer,valid_data)

    #test_data_tokenized,test_attention_mask   = tokenize(tokenizer,test_data)
    
    if args.do_train:
        # Create data set for training
        train_dataset = CreateDataset(train_data_tokenized,train_attention_mask,train_labels)
        valid_dataset = CreateDataset(valid_data_tokenized,valid_attention_mask,valid_labels)

        max_epochs = args.num_train_epochs
        accumulation_steps = args.gradient_accumulation_steps
        evaluation_steps = args.gradient_accumulation_steps*4
        train_batch_size = args.train_batch_size
        valid_batch_size = args.eval_batch_size

        train_dataLoader = generateDataLoader(train_dataset,train_batch_size)
        valid_dataLoader = generateDataLoader(valid_dataset,valid_batch_size)

        print("Generated number of training batches:%d" %len(train_dataLoader))
        print("Generated number of validation batches:%d" %len(valid_dataLoader))
        print("Number of training steps:",len(train_dataLoader)*max_epochs/accumulation_steps)
        print("Number of validation steps:",len(train_dataLoader)*max_epochs/evaluation_steps)

        #### Model Initialization and Training

        # Set the prediction threshold for classifying sarcasm
        PREDICTION_THRESHOLD=0.8
        start = time.time()
        # Initialize the model
        model,optimizer,scheduler,criterion,EPOCHS = bert_model1.initialize_model_for_training(len(train_dataLoader),
                                                            EPOCHS=max_epochs,
                                                            accumulation_steps=accumulation_steps)
        # Train the model
        trained_model,training_stats,validation_stats=bert_model1.run_training(model,train_dataLoader,valid_dataLoader,
                                                        optimizer=optimizer,scheduler=scheduler,criterion=criterion,
                                                        pred_thres=PREDICTION_THRESHOLD,EPOCHS=EPOCHS,
                                                        accumulation_steps=accumulation_steps,evaluation_steps=evaluation_steps,
                                                        logdir=logdir)
        print("Training Time:%0.5f seconds" %(time.time()-start))

if __name__ == "__main__":
    main()

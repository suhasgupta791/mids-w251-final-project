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

# Import custom model classes
from models.Transformers.models import *


# Import Models
from models.Transformers.BertModel import *

# Import utils
from utils.utils import *

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
    parser.add_argument("--train_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The input training file name. Must be provided if --do_train is set.")
   
    parser.add_argument("--trained_model_path",
                        default=None,
                        type=str,
                        required=False,
                        help="full path to trained model when running predictions or resuming training.")
                        
    parser.add_argument("--test_file",
                        default=None,
                        type=str,
                        required=False,
                        help="The input test file name. Must be provided if --do_eval is set.")
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
                        help="Whether to run model in predict mode.")
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
    parser.add_argument("--num_warmup_steps",
                        default=15,
                        type=float,
                        help="Number of steps to perform linear learning rate warmup for"
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
    parser.add_argument('--cross_validation_interval',
                        type=int,
                        default=40,
                        help="Number of training steps to accumulate before performing a cross validation.")
    parser.add_argument('--checkpoint_interval',
                        type=int,
                        default=40,
                        help="Number of steps before saving model checkpoint.")
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
        if device =='cuda':
            from apex import amp
            torch.cuda.empty_cache()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        from apex import amp
        torch.cuda.empty_cache()
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

    # Convert TF checkpoint to pytorch checkpoint and then use as input to class object
    if args.bert_model_path:
        bert_model_path = args.bert_model_path

    # Load data
    data_path = args.data_dir
    if args.do_train:
        if not args.train_file:
            raise ValueError("Training file name must be specified if --do_train is set")
        train_file_path = args.data_dir+"/"+args.train_file
        all_train_df = pd.read_csv(train_file_path)
        #test_df = pd.read_csv(test_file_name)

        # Create a train, valid split
        train_data,train_labels,train_num_excl,valid_data,valid_labels,valid_num_excl = createTestTrainSplit(all_train_df, test_size=0.2,seed=seed)

        train_size,valid_size= len(train_data),len(valid_data)
        print(train_size,valid_size)

        # Create a model object
        bert_model1=Bert_Model(train_df=train_data,
                          bert_model_name=bert_model_name,
                          bert_model_path=bert_model_path,
                          tokenizer=tokenizer,
                          max_seq_length=max_seq_len)

        print("--Tokenizing--")
        train_data_tokenized,train_attention_mask = tokenize(tokenizer,train_data,max_seq_len)
        valid_data_tokenized,valid_attention_mask = tokenize(tokenizer,valid_data,max_seq_len)

        #test_data_tokenized,test_attention_mask   = tokenize(tokenizer,test_data,max_seq_len)
    
        # Create data set for training
        train_dataset = CreateDataset(train_data_tokenized,train_attention_mask,train_labels,train_num_excl)
        valid_dataset = CreateDataset(valid_data_tokenized,valid_attention_mask,valid_labels,valid_num_excl)

        max_epochs = args.num_train_epochs
        accumulation_steps = args.gradient_accumulation_steps
        evaluation_steps = args.cross_validation_interval
        train_batch_size = args.train_batch_size
        valid_batch_size = args.eval_batch_size

        train_dataLoader = generateDataLoader(train_dataset,train_batch_size)
        valid_dataLoader = generateDataLoader(valid_dataset,valid_batch_size)

        print("Generated number of training batches:%d" %len(train_dataLoader))
        print("Generated number of validation batches:%d" %len(valid_dataLoader))
        print("Number of training steps:",len(train_dataLoader)*max_epochs/accumulation_steps)
        print("Number of validation steps:",len(train_dataLoader)*max_epochs/evaluation_steps)

        #### Model Initialization and Training
        start = time.time()

        if args.learning_rate:
            learning_rate = args.learning_rate
        else:
            learning_rate = 2e-5

        # Initialize the model
        model,optimizer,scheduler,criterion,EPOCHS = bert_model1.initialize_model_for_training(len(train_dataLoader),
                                                            EPOCHS=max_epochs,
                                                            accumulation_steps=accumulation_steps,lr=learning_rate,num_warmup_steps=args.num_warmup_steps)
        # Train the model
        trained_model,training_stats,validation_stats=bert_model1.run_training(model,train_dataLoader,valid_dataLoader,
                                                        optimizer=optimizer,scheduler=scheduler,criterion=criterion,
                                                        output_path=args.output_dir,
                                                        pred_thres=PREDICTION_THRESHOLD,EPOCHS=EPOCHS,
                                                        accumulation_steps=accumulation_steps,evaluation_steps=evaluation_steps,
                                                        checkpoint_interval=args.checkpoint_interval,
                                                        logdir=logdir)
        print("Training Time:%0.5f seconds" %(time.time()-start))
        
        # Save the model
        torch.save(trained_model,args.output_dir+"/trained_model.pt")

        # Load and check model 
        model = Bert_Model(train_df=train_data,
                          bert_model_name=bert_model_name,
                          bert_model_path=bert_model_path,
                          tokenizer=tokenizer,
                          max_seq_length=max_seq_len)
        model = torch.load(args.output_dir+"/trained_model.pt")
        model.eval()
        print(model)
    print(args.test_file)
    if args.do_eval:
        if not args.trained_model_path:
            raise ValueError("Trained model path must be provided when running predictions")
        if not args.test_file:
            raise ValueError("Test file name must be provided when running predictions")
               
        test_df = pd.read_csv(args.data_dir+"/"+args.test_file)
        test_data,test_labels,test_num_excl,_,_,_ = createTestTrainSplit(test_df, test_size=0,seed=seed)
        print(len(test_data))
        print("--Tokenizing--")
        test_data_tokenized,test_attention_mask = tokenize(tokenizer,test_data,max_seq_len)
        
        # Create a model object
        bert_model1=Bert_Model(train_df=test_data,
                          bert_model_name=bert_model_name,
                          bert_model_path=bert_model_path,
                          tokenizer=tokenizer,
                          max_seq_length=max_seq_len)

        # Create the dataset object
        test_dataset = CreateDataset(test_data_tokenized,test_attention_mask,test_labels,test_num_excl)
        #Create the data loader object
        test_dataLoader = generateDataLoader(test_dataset,args.eval_batch_size)
        print("Generated number of test batches:%d" %len(test_dataLoader))
        
        
        # Initialize the model
        model,optimizer,scheduler,criterion,EPOCHS = bert_model1.initialize_model(len(test_dataLoader),
                                                                loadfromCheckpoint=False,evalMode=True,model_checkpoint=args.trained_model_path)
        
        # Run predictions
        
        prediction_set,avg_accuracy,avg_f1,avg_precision,avg_recall=bert_model1.run_predict(model,test_dataLoader,args.eval_batch_size,global_step=1,criterion=criterion)
        #print(predictions)
        print("Accuracy:%0.5f | F1 Score:%0.5f | Precision:%0.5f | Recall: %0.5f" %(avg_accuracy,avg_f1,avg_precision,avg_recall))
        np.savetxt(args.output_dir+"/"+"predicted_labels",prediction_set,delimiter=',')
        
        
if __name__ == "__main__":
    main()

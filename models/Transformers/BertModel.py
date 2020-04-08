#!/usr/bin/env python
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
import warnings
warnings.filterwarnings(action='once')
import pickle
import shutil
import time
import matplotlib.pyplot as plt
import tensorflow as tf

# Import transformers specific packages
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import  BertForSequenceClassification, BertForTokenClassification
from transformers import AdamW,get_linear_schedule_with_warmup, pipeline

# Import package for data parallelism to train on multi-GPU machines
from models.Transformers.parallel import DataParallelModel, DataParallelCriterion


# Check if cuda is available
# Set the device and empty cache
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device =='cuda':
    from apex import amp
    torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True

# Class for model training and inference
class Bert_Model():
    def __init__(self,train_df,bert_model_name,bert_model_path,
                tokenizer,test_df=None,
                max_seq_length=128,seed=1234):
        
        if max_seq_length > tokenizer.max_model_input_sizes[bert_model_name]:
            print("Max sequence length specified > 512!!... resetting to 128")
            print("If you don't want this then set max_seq_length to <= 512")
            self._MAX_SEQUENCE_LENGTH = 128
        else:
            self._MAX_SEQUENCE_LENGTH = max_seq_length
        self._SEED = seed
        self._WORK_DIR = "/root/models/Tranformer_based/workingdir/"
        self._bert_model_path=bert_model_path
        self._bert_model_name=bert_model_name
        self._train_data=train_df
        if test_df:
            self._test_size=test_df
        else:
            self._test_size=0
        self._tokenizer = tokenizer
        self._training_stats = []

    def tokenize(self,text_array):
        ''' Returns tokenized IDs and attention mask
        The transformers encode_plus method returns the following:
        {
        input_ids: list[int],
        token_type_ids: list[int] if return_token_type_ids is True (default)
        attention_mask: list[int] if return_attention_mask is True (default)
        overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
        num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
        special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
        }'''
        all_tokens=[]
        all_attention_mask=[]
        for i,text in enumerate(tqdm(text_array)):
            encoded = self._tokenizer.encode_plus(
                           text, 
                           add_special_tokens=True,
                           max_length=self._MAX_SEQUENCE_LENGTH,
                           pad_to_max_length=True)
            tokens = torch.tensor(encoded['input_ids'])
            attention_mask = torch.tensor(encoded['attention_mask'])
            all_tokens.append(tokens)
            all_attention_mask.append(attention_mask)
        return all_tokens,all_attention_mask
     
    def initialize_model_for_training(self,dataset_len,EPOCHS=1,model_seed=21000,lr=2e-5,batch_size=32,
                                      accumulation_steps=2):
        # Setup model parameters
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        torch.cuda.manual_seed(model_seed)
        torch.backends.cudnn.deterministic = True

        # Empty cache
        torch.cuda.empty_cache()
        model = BertForSequenceClassification.from_pretrained(self._bert_model_path,
                                                              cache_dir=None,
                                                              num_labels=2,
                                                              output_attentions = False, 
                                                              output_hidden_states = False)
        model = model.to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = int(EPOCHS*dataset_len/batch_size/accumulation_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr,eps=1e-8,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        if device == 'cuda' :
            model, optimizer = amp.initialize(model,optimizer,opt_level="O1",verbosity=0)
        ### Parallel GPU processing
        #model = DataParallelModel(model) # using balanced data parallalism script
        model = torch.nn.DataParallel(model) # using native pytorch 	
        criterion = nn.CrossEntropyLoss()
        model = model.train()
        model.zero_grad()
        optimizer.zero_grad()
        return model,optimizer,scheduler,criterion,EPOCHS
    
    def run_training(self,model,train_dataLoader,valid_dataLoader,optimizer,scheduler,criterion,
                     EPOCHS=1,batch_size=32,accumulation_steps=20,evaluation_steps=80,pred_thres=0.5,
                     logdir='./logs'):
        # Data Structure for training statistics
        training_stats=[]
        validation_stats=[]
        
        tr_loss = 0.
        tr_accuracy = 0.
        tr_auc = 0.
         
        tq = tqdm(range(EPOCHS),total=EPOCHS,leave=False)
        global_step = 0
        for epoch in tq:
            print("--Training--")
            tk0 = tqdm(enumerate(train_dataLoader),total=len(train_dataLoader),leave=True)
            for step,(x_batch,attn_mask,y_batch) in tk0:
                outputs = model(x_batch.to(device), 
                                token_type_ids=None, 
                                attention_mask=attn_mask.to(device), 
                                labels=y_batch.to(device))
                y_pred = outputs[1]

		# Apply the additional layers 
		
                
                # Parallel GPU processing
                #parallel_loss_criterion = DataParallelCriterion(criterion)

                loss = criterion(y_pred,y_batch.to(device)) / accumulation_steps # when using torch data parallel 
                loss = loss.mean()  # Mean the loss from multiple GPUs
                #loss = parallel_loss_criterion(y_pred,y_batch.to(device))/accumulation_steps # when using balanced data parallel script
                
                if device == 'cuda':
                    with amp.scale_loss(loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tk0.set_postfix(step=global_step+1,loss=loss.item(),auc=tr_auc,accuracy=tr_accuracy) # display running loss
    
                tr_loss += loss.item()
                acc = torch.mean(((torch.sigmoid(y_pred[:,1])>pred_thres) == (y_batch>pred_thres).to(device)).to(torch.float)).item()/len(train_dataLoader)
                #auc = self.compute_auc_score(y_pred[:,1],y_batch.to(device))  # ROC AUC score
                auc = self.compute_auc_score(y_pred.detach().cpu().numpy(),y_batch.cpu().numpy())  # ROC AUC score

                tr_accuracy += acc
                tr_auc += auc

                if (step+1) % accumulation_steps == 0:          # Wait for several backward steps
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the norm to 1.0 to prevent exploding gradients
                    optimizer.step()                            # Now we can do an optimizer step
                    scheduler.step()
                    model.zero_grad()
                    global_step+=1
                    training_stats.append(
                            {
                                'step': global_step,
                                'train_loss': tr_loss/global_step,
                                'train_acc': tr_accuracy/global_step,
                                'train_auc': tr_auc/global_step,
                            })
                    # Write training stats to tensorboard
                    self.summaryWriter("train",loss.item(),acc,auc,global_step+1,logdir)            
                    # Run evaluation when no gradients are accumulated
                    if (step+1) % evaluation_steps ==0:
                        print("--I-- Running Validation")
                        eval_loss,eval_accuracy,eval_auc=self.run_eval(model,valid_dataLoader,global_step,criterion)
                        validation_stats.append(
                            {
                                'step': global_step,
                                'valid_loss': eval_loss,
                                'valid accuracy': eval_accuracy,
                                'valid auc score': eval_auc,
                            })
                        # Write training stats to tensorboard
                        self.summaryWriter("eval",eval_loss,eval_accuracy,eval_auc,global_step,logdir)
            tq.set_postfix(train_loss=tr_loss/(global_step+1),train_accuracy=tr_accuracy/(global_step+1),train_auc=tr_auc/(global_step+1))
        return model,training_stats,validation_stats
    
    def run_eval(self,model,valid_dataLoader,global_step,criterion):
        avg_loss = 0.
        eval_accuracy = 0.
        eval_loss = 0.
        eval_auc = 0.
        nb_eval_steps = 0
        tk0 = tqdm(enumerate(valid_dataLoader),total=len(valid_dataLoader),leave=True)
        for step,(x_batch, attn_mask,y_batch) in tk0:
            model.eval()
            with torch.no_grad():
                outputs = model(x_batch.to(device), 
                                token_type_ids=None, 
                                attention_mask=attn_mask.to(device), 
                                labels=y_batch.to(device))
            loss, y_pred = outputs
            loss = criterion(y_pred,y_batch.to(device))
            y_pred = y_pred.detach().cpu().numpy()
            label_ids = y_batch.to('cpu').numpy()

            tmp_eval_accuracy = self.flat_accuracy(y_pred, label_ids)
            tmp_eval_auc = self.compute_auc_score(y_pred, label_ids) ## ROC AUC Score

            loss = criterion(y_pred,y_batch.to(device)) / accumulation_steps # when using torch data parallel 
            loss = loss.mean()  # Mean the loss from multiple GPUs

            # Accumulate the total accuracy.
            eval_loss += loss/len(valid_dataLoader)
            eval_accuracy += tmp_eval_accuracy/len(valid_dataLoader)
            eval_auc += tmp_eval_auc
            nb_eval_steps += 1
        avg_loss = eval_loss/nb_eval_steps
        avg_accuracy = eval_accuracy/nb_eval_steps
        avg_auc = eval_auc/nb_eval_steps
        tk0.set_postfix(step=global_step,avg_loss=avg_loss,avg_accuracy=avg_accuracy,avg_auc=avg_auc)
        return avg_loss,avg_accuracy,avg_auc

     # Function to calculate the accuracy of predictions vs labels
    def flat_accuracy(self,preds,labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)

    def compute_auc_score(self,preds,labels):
        auc_score = roc_auc_score(labels.flatten(),preds[:,1].flatten())
        return auc_score
    
    def summaryWriter(self,name,loss,acc,auc,n_iter,logdir):
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter(logdir)
        writer.add_scalar('Loss/'+name,loss,n_iter)
        writer.add_scalar('Accuracy/'+name,acc,n_iter)
        writer.add_scalar('ROC AUC Score/'+name,auc,n_iter)
        writer.close()

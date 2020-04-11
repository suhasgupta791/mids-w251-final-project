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
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
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

# Import custom models
from models.Transformers.CNNModel2 import CNN
from models.Transformers.models import BertFC, BertSeq,BertFCWithExclamation, BertSeq,BertFCWithExclamation2
from models.Transformers.BertCustom import BertCustom

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
        
        ## Construct Bert model with pre trained weights

        # Bert Only Model
#        model = BertForSequenceClassification.from_pretrained(self._bert_model_path,
#                                                              cache_dir=None,
#                                                              num_labels=768,
#                                                              output_attentions = False,
#                                                              output_hidden_states = False)
                                                              
        ## Bert + CNN Model 
        #embed_num = self._MAX_SEQUENCE_LENGTH
        embed_dim = 768
        class_num = 2
        #kernel_num = 3
        #kernel_sizes = [10,15,20]
        dropout = 0.3
        #static = True

        #model = CNN(batch_size,self._bert_model_path,
                    #embed_num=embed_num,
                    #embed_dim=embed_dim,
                    #class_num=class_num,
                    #kernel_num=kernel_num,
                    #kernel_sizes=kernel_sizes,
                    #dropout=dropout,
                    #static=static,
                    #)

        #model = BertFC(batch_size,self._bert_model_path,
                    #embed_dim=embed_dim,
                    #class_num=class_num,
                    #dropout=dropout
                    #)

        #model = BertSeq(batch_size,self._bert_model_path,
                    #embed_dim=embed_dim,
                    #class_num=class_num,
                    #dropout=dropout
                    #)
        
        model = BertFCWithExclamation(self._bert_model_path,
                    embed_dim=embed_dim,
                    class_num=class_num,
                    dropout=dropout
                    )
        model = BertFCWithExclamation2(self._bert_model_path,
                    embed_dim=embed_dim,
                    class_num=class_num,
                    dropout=dropout
                    )


        
        model = model.to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = int(EPOCHS*dataset_len/batch_size/accumulation_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr,eps=1e-8,correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=10,num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
        if device == 'cuda' :
            model, optimizer = amp.initialize(model,optimizer,opt_level="O1",verbosity=0)
        ### Parallel GPU processing
        #model = DataParallelModel(model) # using balanced data parallalism script
        model = torch.nn.DataParallel(model) # using native pytorch 	
#        criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        model = model.train()
        model.zero_grad()
        optimizer.zero_grad()
        return model,optimizer,scheduler,criterion,EPOCHS
    
    def run_training(self,model,train_dataLoader,valid_dataLoader,optimizer,scheduler,criterion,
                     EPOCHS=1,tr_batch_size=32,accumulation_steps=20,evaluation_steps=80,pred_thres=0.5,
                     logdir='./logs'):
        # Data Structure for training statistics
        training_stats=[]
        validation_stats=[]
        
        tr_loss = 0.
        tr_accuracy = 0.
        tr_auc = 0.
        tr_f1 = 0.
         
        tq = tqdm(range(EPOCHS),total=EPOCHS,leave=False)
        global_step = 0
        for epoch in tq:
            print("--Training--")
            tk0 = tqdm(enumerate(train_dataLoader),total=len(train_dataLoader),leave=True)
            for step,(x_batch,attn_mask,y_batch,num_excl_batch) in tk0:
                #num_excl_batch = torch.tensor(num_excl_batch,dtype=torch.float)
                num_excl_batch = num_excl_batch.type(torch.float)
                outputs = model(x_batch.to(device),num_excl_batch.to(device),
                                token_type_ids=None, 
                                attention_mask=attn_mask.to(device), 
                                labels=y_batch.to(device))
                y_pred = outputs
                predicted_probs,predicted_labels = self.classifyWithThreshold(y_pred,y_batch)

                # Parallel GPU processing

                # Loss
                loss = self.computeLoss(criterion,predicted_probs,y_batch)  
                #loss = parallel_loss_criterion(y_pred,y_batch.to(device))/accumulation_steps # when using balanced data parallel script

                if device == 'cuda':
                    with amp.scale_loss(loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()/accumulation_steps  # accumulate the global loss (divide by gloabal step to reflect moving average)
                
                # Accuracy
                if step>0:
                    acc = torch.mean((predicted_labels == y_batch.to(device)).to(torch.float)).item()  # accuracy for the whole batch
                else:
                    acc = 0
                tr_accuracy += acc/accumulation_steps
                # AUC Score
                c_report_dict = self.class_report(predicted_labels,y_batch)
#                print(self.conf_matrix(predicted_labels,y_batch))
                f1_score = c_report_dict['1']['f1-score']
                tr_f1 +=f1_score/accumulation_steps
                auc = self.compute_auc_score(y_pred[:,1],y_batch)
                tr_auc += auc/accumulation_steps

                tk0.set_postfix(step=step+1,forward=(step+1)%accumulation_steps,loss=loss.item(),accuracy=acc) # display running backward loss

                if (step+1) % accumulation_steps == 0:          # Wait for several backward steps
                    # Write training stats to tensorboard
                    self.summaryWriter("train",tr_loss,tr_accuracy,tr_auc,tr_f1,global_step,logdir)
 
                    # Zero out the evaluation metrics after several backward steps
                    tr_accuracy = 0
                    tr_loss = 0
                    tr_auc = 0
                    tr_f1 = 0 

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the norm to 1.0 to prevent exploding gradients
                    optimizer.step()                            # Now we can do an optimizer step
                    scheduler.step()
                    model.zero_grad()
                    global_step+=1  # increment forward step count
                    training_stats.append(
                            {
                                'step': global_step,
                                'train_loss': tr_loss/global_step,
                                'train_acc': tr_accuracy/global_step,
                                'train_auc': tr_auc/global_step,
                            })
                    #Run evaluation after several forward passes (determined by evaluation_steps)
                    if (step+1) % evaluation_steps ==0:
                        print("--I-- Running Validation")
                        print(step,evaluation_steps)
                        eval_loss,eval_accuracy,eval_auc,eval_f1=self.run_eval(model,valid_dataLoader,global_step,criterion)
                        validation_stats.append(
                            {
                                'step': global_step,
                                'valid_loss': eval_loss,
                                'valid accuracy': eval_accuracy,
                                'valid auc score': eval_auc,
                            })
                        # Write training stats to tensorboard
                        self.summaryWriter("eval",eval_loss,eval_accuracy,eval_auc,eval_f1,global_step,logdir)
            tq.set_postfix(train_loss=tr_loss,train_accuracy=tr_accuracy,train_auc=tr_auc,leave=False)
        return model,training_stats,validation_stats
    
    def run_eval(self,model,valid_dataLoader,global_step,criterion):
        avg_loss = 0.
        eval_accuracy = 0.
        eval_loss = 0.
        eval_auc = 0.
        eval_f1 = 0.
        nb_eval_steps = 0
        tk0 = tqdm(enumerate(valid_dataLoader),total=len(valid_dataLoader),leave=True)
        for step,(x_batch, attn_mask,y_batch,num_excl_batch) in tk0:
            model.eval()
            with torch.no_grad():
                #num_excl_batch = torch.tensor(num_excl_batch,dtype=torch.float)
                num_excl_batch = num_excl_batch.type(torch.float)
                outputs = model(x_batch.to(device),num_excl_batch.to(device), 
                                token_type_ids=None, 
                                attention_mask=attn_mask.to(device), 
                                labels=y_batch.to(device))
            #loss, y_pred = outputs
            y_pred = outputs

            predicted_probs,predicted_labels = self.classifyWithThreshold(y_pred,y_batch)

            # Loss
            loss = self.computeLoss(criterion,predicted_probs,y_batch)  
            eval_loss += loss.item()
            
            # Accuracy
            # Accuracy
            eval_accuracy += torch.mean((predicted_labels == y_batch.to(device)).to(torch.float)).item()  # accuracy for the whole batch
            
            # AUC Score
            auc = self.compute_auc_score(y_pred[:,1],y_batch.to(device))
            eval_auc += auc

            # F1 Score 
            c_report_dict = self.class_report(predicted_labels,y_batch)
            f1_score = c_report_dict['1']['f1-score']
            eval_f1 += f1_score

#            tmp_eval_auc = self.compute_auc_score(predicted_labels, label_ids) ## ROC AUC Score
            
            # Increment total eval step count
            nb_eval_steps += 1
        
        # Normalize to the number of steps
        avg_loss = eval_loss/nb_eval_steps
        avg_accuracy = eval_accuracy/nb_eval_steps
        avg_auc = eval_auc/nb_eval_steps
        avg_f1 = eval_f1/nb_eval_steps
        
        tk0.set_postfix(step=global_step,avg_loss=avg_loss,avg_accuracy=avg_accuracy,avg_auc=avg_auc)
        return avg_loss,avg_accuracy,avg_auc,avg_f1

    def computeLoss(self,criterion,predicted_probs,y_true):
        loss = criterion(predicted_probs,y_true.type(torch.float).to(device)) # when using torch data parallel
        loss = loss.mean()  # Mean the loss from multiple GPUs and to take care of batch size
        return loss

     # Function to calculate the accuracy of predictions vs labels
    def flat_accuracy(self,preds,labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)

    def compute_auc_score(self,preds,labels):
        labels = labels.cpu().numpy()
        preds = preds.detach().cpu().numpy()
        auc_score = roc_auc_score(labels.flatten(),preds.flatten())
        return auc_score
    
    def class_report(self,preds,labels):
        c_report = classification_report(labels.flatten(),preds.detach().cpu().numpy().flatten(),output_dict=True,zero_division=0)
        return c_report
    
    def conf_matrix(self,preds,labels):
        conf_matrix = confusion_matrix(labels.flatten(),preds.detach().cpu().numpy().flatten())
        return conf_matrix

    def classifyWithThreshold(self,preds,labels):
        pass
        pred_after_sigmoid = torch.sigmoid(preds) # Apply the sigmoid to the logits from output of Bert
        pred_probs,pred_classes = torch.max(pred_after_sigmoid,dim=-1)
        return pred_probs,pred_classes
    
    def summaryWriter(self,name,loss,acc,auc,f1_score,n_iter,logdir):
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter(logdir)
        writer.add_scalar('Loss/'+name,loss,n_iter)
        writer.add_scalar('Accuracy/'+name,acc,n_iter)
        writer.add_scalar('ROC_AUC_Score/'+name,auc,n_iter)
        writer.add_scalar('F1_Score/'+name,f1_score,n_iter)

        writer.close()

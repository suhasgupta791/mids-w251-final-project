#!/usr/bin/python3

import sys, getopt
import praw
from praw.models import MoreComments
import json

import itertools
import random
import numpy as np
import os
import torch
import tqdm

import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler

from transformers import BertModel, BertConfig
from transformers import  BertForSequenceClassification, BertForTokenClassification
from transformers import AdamW,get_linear_schedule_with_warmup, pipeline

# Import package for data parallelism to train on multi-GPU machines
from models.Transformers.parallel import DataParallelModel, DataParallelCriterion

# Import custom models
from models.Transformers.CNNModel2 import CNN
from models.Transformers.models import *
from models.Transformers.BertCustom import BertCustom

from utils.utils import *

class CreateDataset(Dataset):
    def __init__(self,data,atten_mask,labels,num_excl):
        self._dataset = [[data[i],atten_mask[i],labels[i],num_excl[i]] for i in range(0,len(data))]
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self,idx):
        return self._dataset[idx]

def tokenize(text):
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

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    encoded = tokenizer.encode_plus(
                        text, 
                        add_special_tokens=True,
                        max_length=128,
                        pad_to_max_length=True)

    tokens = torch.tensor(encoded['input_ids'])
    attention_mask = torch.tensor(encoded['attention_mask'])
    return tokens,attention_mask

def classifyWithThreshold(preds,labels):
    pass
    pred_after_sigmoid = torch.sigmoid(preds) # Apply the sigmoid to the logits from output of Bert
    pred_probs,pred_classes = torch.max(pred_after_sigmoid,dim=-1)
    return pred_probs,pred_classes

def run_eval(input, model, device):
    for step, (x_batch, attn_mask,y_batch,num_excl_batch) in enumerate(input):
        model.eval()
        with torch.no_grad():
            num_excl_batch = num_excl_batch.type(torch.float)
            outputs = model(x_batch.to(device), num_excl_batch.to(device),token_type_ids=None,attention_mask=attn_mask.to(device),labels=y_batch.to(device))

        y_pred = outputs
        predicted_prob,predicted_label = classifyWithThreshold(y_pred,y_batch)

    return predicted_prob,predicted_label

def predict_comment(device, model, textinfo):
    label=0
    label_text="NON-SARCASTIC"
    exclamationNum = textinfo.count("!")

    if (textinfo.find("\s") != -1):
        textinfo.replace("\s", "")
        label=1
        label_text="SARCASTIC"

    tokens, attention_mask = tokenize(textinfo)
    input = CreateDataset([tokens], [attention_mask], [label], [exclamationNum])
    new_input = generateDataLoader(input, 1)

    predicted_prob,predicted_label = run_eval(new_input, model, device)
    predicted_prob = predicted_prob.item()
    predicted_text = "SARCASTIC" if predicted_label.item() == 1 else "NOT SARCASTIC"
    predicted_prob = predicted_prob if predicted_label.item() == 1 else 1.0-predicted_prob

    print("OUTPUT: For this " + label_text + " comment, we predicted it is " + predicted_text + " with {:.2f}% confidence".format(predicted_prob))
    print("COMMENT: " + textinfo + "\n-----------------------------------------------------------")

def main(argv):
    subreddit_name = "politics"
    numComments = 3
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:",["subreddit="])
    except getopt.GetoptError:
        print('test.py -s <subreddit>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -s <subreddit> -n <number of comments>')
            sys.exit()
        elif opt in ("-s", "--subreddit"):
            subreddit = arg
        elif opt in ("-n", "--iterations"):
            numComments = arg
   
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    model = torch.load("/root/trained_model.pt")

    reddit = praw.Reddit(client_id='kH0hImI7NwEKoQ',
                         client_secret='VWc4D0SPuZSALQ267bfILDhzSMk',
                         user_agent='script:ucb-w251-project:v0.1 (by u/enex)')

    subreddit = reddit.subreddit(subreddit_name)

    for submission in subreddit.hot(limit=numComments):
        submission.comments.replace_more()
        commentlist = []

    for comment in submission.comments:
        predict_comment(device, model, comment.body)

if __name__ == "__main__":
   main(sys.argv[1:])
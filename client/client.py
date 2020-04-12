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

    print("token1")

    encoded = tokenizer.encode_plus(
                        text, 
                        add_special_tokens=True,
                        max_length=128,
                        pad_to_max_length=True)

    tokens = torch.tensor(encoded['input_ids'])
    attention_mask = torch.tensor(encoded['attention_mask'])
    print("token3")
    return tokens,attention_mask

def run_eval(tokens, attn_mask, label, exclamationCount, model):
    print("eval1")
   
    model.eval()
    with torch.no_grad():
        exclamationCount = exclamationCount.type(torch.float)
        outputs = model(tokens.to(device), exclamationCount.to(device),
                        token_type_ids=None,
                        attention_mask=attn_mask.to(device),
                        labels=label.to(device))

    y_pred = outputs
    print("eval2")

    predicted_prob,predicted_label = BertModel.classifyWithThreshold(y_pred,label)

    return predicted_prob,predicted_label

def predict_comment(textinfo):
    label=0
    label_text="NOT SARCASTIC"
    exclamationNum = textinfo.count("!")

    # commentlist.append(textinfo)
    if (textinfo.find("\s") != -1):
        print("found!")
        textinfo.replace("\s", "")
        label=1
        label_text="SARCASTIC"

    print("predict1")
    tokens, attention_mask = tokenize(textinfo)
    # input = [tokens, attention_mask, sarcastic, exclamationNum]
    predicted_prob,predicted_label = run_eval(tokens, attention_mask, sarcastic, exclamationNum, model)
    print("predict2")

    if (predicted_label == 1):
        predicted_text = "SARCASTIC"
    else:
        predicted_text = "NOT SARCASTIC"

    print("OUTPUT: For " + label_text + " comment, predicted it is " + predicted_text + " with " + predicted_prob + "% confidence")
    print("COMMENT: " + textinfo + "\n")

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
            print ('test.py -s <subreddit>')
            sys.exit()
        elif opt in ("-s", "--subreddit"):
            subreddit = arg
        elif opt in ("-n", "--iterations"):
            numComments = arg
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("/root/trained_model.pt")
    print("part1")

    reddit = praw.Reddit(client_id='kH0hImI7NwEKoQ',
                         client_secret='VWc4D0SPuZSALQ267bfILDhzSMk',
                         user_agent='script:ucb-w251-project:v0.1 (by u/enex)')

    subreddit = reddit.subreddit(subreddit_name)
    print("part2")

    for submission in subreddit.hot(limit=numComments):
        submission.comments.replace_more()
        commentlist = []
        print("part3")

    for comment in submission.comments:
        predict_comment(comment.body)

if __name__ == "__main__":
   main(sys.argv[1:])
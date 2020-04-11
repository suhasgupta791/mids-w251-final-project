#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import  BertForSequenceClassification, BertForTokenClassification
from transformers import BertModel, BertConfig

class BertFC(nn.Module):
    def __init__(self, batch_size,bert_model_path,embed_dim, class_num, dropout):
        super(BertFC,self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_path,
                                                                  cache_dir=None,
                                                                  num_labels=768,
                                                                  output_attentions = False,
                                                                  output_hidden_states = True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim,int(embed_dim/2))
        self.fc2 = nn.Linear(int(embed_dim/2),int(embed_dim/4))
        self.fc3 = nn.Linear(int(embed_dim/4),class_num)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None):

        logits,hidden_states = self.bert(input_ids=x,token_type_ids=token_type_ids,attention_mask=attention_mask)
        outputs = logits     
        outputs = self.dropout(outputs)
        outputs = self.fc1(outputs)  
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)  
        outputs = self.dropout(outputs)
        outputs = self.fc3(outputs) 
        return outputs

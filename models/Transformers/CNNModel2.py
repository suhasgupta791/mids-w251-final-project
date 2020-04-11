#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import  BertForSequenceClassification, BertForTokenClassification
from transformers import BertModel, BertConfig

class CNN(nn.Module):
    def __init__(self, batch_size,bert_model_path, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
        super(CNN,self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_path,
                                                                  cache_dir=None,
                                                                  num_labels=768,
                                                                  output_attentions = False,
                                                                  output_hidden_states = True)
        V = embed_num
        D = embed_dim
        C = class_num
        Co = kernel_num
        Ks = kernel_sizes
        self.Ks = Ks
        
        self.static = static
        self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None):

        logits,hidden_states = self.bert(input_ids=x,token_type_ids=token_type_ids,attention_mask=attention_mask)
        outputs = hidden_states[0]
        outputs = outputs.unsqueeze(1)
        outputs = [F.relu(conv(outputs)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        outputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in outputs]  # [(N, Co), ...]*len(Ks)
        shapes = [outputs[x].shape for x in range(len(self.Ks))]
        outputs = torch.cat(outputs, 1)
        outputs = self.dropout(outputs)  # (N, len(Ks)*Co)
        outputs = self.fc1(outputs)  # (N, C) 
        outputs = self.sigmoid(outputs)
        return outputs

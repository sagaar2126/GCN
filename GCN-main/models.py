# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:40:02 2021

@author: HP
"""
from torch.nn import Module
from layers import GCNConv
from torch.nn.functional import log_softmax, relu
from torch.nn.functional import dropout

class GCN(Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_ratio = 0.5):
        super(GCN, self).__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.dropout_ratio = dropout_ratio
        
    def forward(self, X, edges):
        output = self.gcn1(X, edges)
        output = relu(output)
        output = dropout(output, self.dropout_ratio, training = self.training)
        output = self.gcn2(output, edges)
        output = log_softmax(output)
        return output
            

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:35:53 2021

@author: HP
"""
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from utils import glorot, zeros

class GCNConv(Module):
    
    def __init__(self, input_dim, output_dim, bias=True):
        super(GCNConv, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = Parameter(torch.FloatTensor(input_dim, output_dim))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    #Initialize weights and bias
    def reset_parameters(self):
        glorot(self.weights)
        zeros(self.bias)
    
    def forward(self, X, edges):
        mult1 = torch.matmul(edges, X)
        output = torch.matmul(mult1, self.weights)
        if self.bias is not None:
            output += self.bias
        return output



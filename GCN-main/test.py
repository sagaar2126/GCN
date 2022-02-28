# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 08:18:24 2021

@author: HP
"""
import argparse
import time
from data_extraction import data_extraction
import numpy as np
import torch
from models import GCN
from torch.nn.functional import nll_loss
from utils import accuracy

#parser
parser = argparse.ArgumentParser(description='Node classification on Cora dataser')
parser.add_argument('-e', '--Epochs', type = int, default = 200, help = 'NUmber of Epochs')
parser.add_argument('-lr', '--Learning_Rate', type = float, default = 0.01, help = 'Learning rate')
parser.add_argument('-hi', '--Hidden', default=16, type = int, help = 'Number of hidden layers')
parser.add_argument('-dp', '--Dropout', default = 0.4, type = float, help= 'Dropout')
parser.add_argument('-wd', '--Weight_Decay', default = 5e-4, type = float, help = 'Weight Decay for optimizer')
args = parser.parse_args()

#Data paths
path = 'data/citeseer/'
dataset = 'citeseer'

#Start timestamp
t0 = time.time() 

#Loading data
features, adjacency, labels, train_idx, val_idx, test_idx = data_extraction()

#Hyperparameters
input_dim = features.shape[1]
hidden_dim = args.Hidden
output_dim = len(np.unique(labels))
learning_rate = args.Learning_Rate
dropout_ratio = args.Dropout
decay = args.Weight_Decay
epochs = args.Epochs

FILE = "/home/sagar/Downloads/GCN-main/model1.pth"

loaded_model = GCN(input_dim, hidden_dim, output_dim, dropout_ratio)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

def test():
    
    #Model to training mode
    loaded_model.eval()
    
    #Testing loss and accuracy
    y_hat = loaded_model(features, adjacency)
    test_loss = nll_loss(y_hat[test_idx], labels[test_idx])
    test_acc = accuracy(labels[test_idx], y_hat[test_idx])
    
    print(f'Test Loss :{test_loss:.4f}, '
          f'Test acc :{test_acc:.4f}')

#Testing    
test()

#End timestamp
print(f'Time Taken :{time.time() - t0}')
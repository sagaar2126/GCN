# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:40:25 2021

@author: HP
"""

import argparse
from data_extraction import data_extraction
from torch.optim import Adam
from models import GCN
import numpy as np
from torch.nn.functional import nll_loss
from utils import accuracy
import time
import copy
import torch
from tqdm import tqdm

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

#Model
model = GCN(input_dim, hidden_dim, output_dim, dropout_ratio)

#Loss and Optimizer
optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = decay)
 

def train():
    
    best_model_wts = copy.deepcopy(model.state_dict()) 
    best_acc = 0.
    epoch_no = 0
    
    for epoch in (range(epochs)):
        
        #Model to training mode
        model.train()
        
        # Forward pass
        y_hat = model(features, adjacency)
        train_loss = nll_loss(y_hat[train_idx], labels[train_idx])
        
        #Backward and optimize
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        #Training accuracy
        train_acc = accuracy(labels[train_idx], y_hat[train_idx])
        
        #Model to evaluation mode
        model.eval()
        
        #Validation loss and accuracy
        val_loss = nll_loss(y_hat[val_idx], labels[val_idx])
        val_acc = accuracy(labels[val_idx], y_hat[val_idx])
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epoch_no = epoch
            
        print(f'Epochs :{epoch+1}/{epochs}, '
              f'Train Loss :{train_loss:.4f}, '
              f'Train_acc :{train_acc:.4f}, '
              f'Val Loss :{val_loss:.4f}, '
              f'Val acc :{val_acc:.4f}')
            
    return best_acc, epoch_no, best_model_wts
    
  
#Training
best_acc, epoch_no, best_model_wts = train()
model.load_state_dict(best_model_wts)

print(f'Best val acc :{best_acc}, Epoch :{epoch_no}')

# #Saving Model
# FILE = "saved_model/model1.pth"
# torch.save(model.state_dict(), FILE)
# print('Model Saved!!')

#End timestamp
print(f'Time Taken :{time.time() - t0}')
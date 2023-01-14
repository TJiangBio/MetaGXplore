# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 21:24:44 2023

@author: Haiyang Jiang
"""

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


class GCN_Net(torch.nn.Module):
    def __init__(self,args):
        super(GCN_Net, self).__init__()
        self.args = args
        self.nhid = args.nhid
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.pooling_ratio = args.pooling_ratio
        self.K = args.K
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        
        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self. num_classes)
    
    def reset_parameters(self):
        
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, x, edge_index, batch):
        
        x = F.relu(self.conv1(x, edge_index))
        
        x = F.relu(self.conv2(x, edge_index))
        
        x = F.relu(self.conv3(x, edge_index))
        
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
                
        x = F.relu(self.lin1(x))
        #print("x",x.shape)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        #print("x",x.shape)
        x = F.log_softmax(self.lin3(x), dim=-1)
       # print("x",x.shape)
        return x








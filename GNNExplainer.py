# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:04:10 2022

@author: Haiyang Jiang
"""

from GNN_network import GCN_Net
import torch
from torch_geometric.data import DataLoader
import argparse
from torch.utils.data import random_split
from torch_scatter import scatter_add
from creat_dataset import MyOwnDataset
from torch_geometric.nn import GNNExplainer
import numpy as np
import pandas as pd



patience = 0

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--epochs', type=int, default=10000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--hops', type=str, default=5,
                    help='number of hops')
parser.add_argument('--n_clusters', type=str, default=2,
                    help='number of clusters')
parser.add_argument('--heads', type=int, default=4)

parser.add_argument('--K', type=int, default=5)
parser.add_argument('--K1', type=int, default=3)
parser.add_argument('--layer', type=int, default=5, help='number of layers.')
parser.add_argument('--bias', default='none', help='bias.')
parser.add_argument('--alfa', type=float, default=0.5)

args = parser.parse_args()
args.device = 'cuda:0'
torch.manual_seed(args.seed)
dataset = MyOwnDataset("E://2022mofometa//MYdata123")
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features
print(args)

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set,validation_set,test_set = random_split(dataset,[num_training,num_val,num_test])

print("dataset", dataset)

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set,batch_size=args.batch_size,shuffle=False)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

model = GCN_Net(args)
model.load_state_dict(torch.load('latest2.pth'))

explainer = GNNExplainer(model, epochs=200,return_type='log_prob')#.to(args.device)


mask = []
sort_edge = []
for data in dataset:
    x, edge_index = data.x, data.edge_index
    node_feat_mask, edge_mask = explainer.explain_graph(x, edge_index)
    mask.append([node_feat_mask.numpy(), edge_mask.numpy(), data.y.numpy()])  
    sort_edge.append(np.argsort(edge_mask.numpy()))                                             

max_sub_graph = pd.read_csv('utils_files//max_sub_graph.csv')

edge_mask_save = "test_result//edge_mask//"
feature_mask_save = "test_result//feature_mask//"
node_weight_save = "test_result//node_weight//"

test_top_edge_str = []
node_weight_list = []
k=0
for mask_ in mask:
    em = mask_[1]
    fm = mask_[0]
    sort_em = np.argsort(em)
    top_edges = sort_em[-50:]
    top_edge_index = edge_index.T.numpy()[top_edges]
    top_edge_str = []
    for i in range(top_edge_index.shape[0]):
        #print(top_edge_index[i,0],top_edge_index[i,1])
        src = max_sub_graph.iloc[top_edge_index[i,0],0]
        dst = max_sub_graph.iloc[top_edge_index[i,1],0]
        top_edge_str.append(np.array([src,dst]).reshape(1,2))
    top_edge_str = np.concatenate(top_edge_str,axis=0)
    np.savetxt(edge_mask_save+'top_edges'+str(k)+'_'+str(int(mask_[-1]))+'.csv',top_edge_str,fmt='%s',delimiter=',')
    np.savetxt(feature_mask_save+'feat_mask'+str(k)+'_'+str(int(mask_[-1]))+'.csv',fm,fmt='%f',delimiter=True)
    
    test_top_edge_str.append(top_edge_str)
    
    num_nodes = x.shape[0]
    row, col = edge_index[0], edge_index[1]
    deg_weight = scatter_add(torch.tensor(em), col, dim=0, dim_size=num_nodes)
    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    #print(deg)
    node_weight = (deg_weight*deg.pow_(-1)).reshape(-1,1)
    node_weight_list.append(node_weight)
    np.savetxt(node_weight_save+'node_weight'+str(k)+'_'+str(int(mask_[-1]))+'.csv',node_weight,fmt='%f',delimiter=True)
    k = k+1


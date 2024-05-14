# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 10:26:22 2022

@author: Haiyang Jiang
"""


from GNN_network import GCN_Net
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import argparse
from torch.utils.data import random_split
import time
from creat_dataset import MyOwnDataset
from sklearn.metrics import f1_score

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
parser.add_argument('--heads', type=int, default=1)

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

def test(model,loader, f1=False):
    model.eval()
    correct = 0.
    loss = 0.
    i=0
    prob_all = []
    label_all = []
    for data in loader:
        data = data.to(args.device)
        #out, A, B, X_num = model(data)
        out = model(data.x, data.edge_index, data.batch)
        #print("out",out)
        pred = out.max(dim=1)[1]
        prob_all.append(pred.cpu().numpy())
        label_all.append(data.y.cpu().numpy())
        #print("pred",pred)
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out,data.y,reduction='sum').item()
    if f1==True:
        return correct / (len(loader.dataset)-i),loss / (len(loader.dataset)-i), f1_score(label_all,prob_all)
    else:
        return correct / (len(loader.dataset)-i),loss / (len(loader.dataset)-i)

def train(model):
    
    val_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    min_loss = 1e10
    patience = 0
    t1 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader):
            #print("data", data)
            data = data.to(args.device)
            out= model(data.x, data.edge_index, data.batch)
            #print("out shape",out.shape, data.y)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        val_acc,val_loss = test(model,val_loader, False)
        val_list.append(val_acc)
        print("Validation loss:{}\taccuracy:{}".format(val_loss,val_acc), 'time: {:.6f}s'.format(time.time() - t1))
        if val_loss < min_loss:
            torch.save(model.state_dict(),'latest2.pth')
            print("Model saved at epoch{}".format(epoch))
            print("################Test accuarcy:{}".format(test_acc))
            min_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            print("Early stop!")
            break 
    return val_list

model1 = GCN_Net(args).to(args.device)

val_list1 = train(model1)

model = GCN_Net(args).to(args.device)
model.load_state_dict(torch.load('latest2.pth'))
test_acc,test_loss, F1_score = test(model,test_loader, True)
print("Test accuarcy:{:.2f}".format(test_acc*100))
print("F1-Score:{:.4f}".format(F1_score))



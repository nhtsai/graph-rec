from process import*
from utils import*
from model import GraphSAGE
from eval import*

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import json
import gzip
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dgl.data

def train(fp="../../data/meta_Electronics.json.gz", epochs=100, hit=True):
    '''
    Processes Data & Trains GraphSAGE Model

    parameters: str Filepath to data(fp), 
                int # Of Epochs (epochs), 
                boolean Compute Hit-Rate (hit)


    returns: tensor Trained embeddings (h),
            GraphSAGE Trained Model (model)

    '''
    #Preprocess
    edges, node_features, node_labels, edges_src, edges_dst = preprocess(fp)

    #Build DGL Graph
    g = build_graph(node_features, node_labels, edges_src, edges_dst)

    #Train-Test Split Graphs
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g = train_test_split(g):

    #Inits
    model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)
    pred = DotPredictor()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    epochs = epochs

    #Training
    for epoch in range(epochs):
        #Forward
        h = model(train_g, train_g.ndata['feat'])
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        
        #Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('In epoch {}, loss: {}'.format(epoch, loss))

    if hit:
        hits = get_hits(edges, h, model(test_pos_g, node_features))
        print(np.mean(hits))
    
    return h, model

if __name__ == '__main__':
    train()



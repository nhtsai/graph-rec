import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(pos_score, neg_score):
    ''' 
    Computes cross entropy loss on edge features
    
    Source: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    '''
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def get_hits(edges_df, h, h_test):
    '''
    Gets list of hits given the parameters below

    parameters: edges list df, 
                h embedings from training, 
                h_test embeddings from test set.
    returns: list of hits
    '''
    hits = []
    edges = edges_df
    for i in range(h.shape[0]):
        true_edges = list(edges[edges.asin == i].also_bought)
        dist = torch.cdist(h_test[[i]], h)
        top_k = torch.topk(dist, k = 500, largest=False)[1]
        hit = 0
        for j in true_edges:
            if j in top_k:
                hit = 1
                break
        hits.append(hit)
    return hits
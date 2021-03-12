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

from utils import getDF

def preprocess(fp):
    '''
    Cleans and process Amazon product metadata

    Parameters: str Filepath (fp)
    Returns: DataFrame edges, 
            Tensor node_features, 
            Tensor node_labels, 
            Tensor edges_src, 
            Tensor edges_dst
    '''
    df = getDF(fp)
    df = df.dropna()
    #Only keep rows with "Also Bought" in the related column
    df.related = df.related.apply(lambda x: x if 'also_bought' in x.keys() else np.nan)
    df = df.dropna()

    #Create also_bought column
    df['also_bought'] = df.related.apply(lambda x: x['also_bought'])

    #Remove all product IDs (ASINs) outside the dataset
    df['also_bought'] = df['also_bought'].apply(lambda x: set(df.asin).intersection(x))
    df['also_bought'] = df['also_bought'].apply(lambda x: list(x) if len(x) > 0 else np.nan)
    df = df.dropna().reset_index(drop=True)

    #Finds Niche Category of each product
    df['niche'] = df.categories.apply(lambda x: str(x).strip(']').split(',')[-1])

    df = df.explode('also_bought')

    all_nodes = list(set(df.asin).intersection(set(df.also_bought)))
    all_nodes = list(set(df.asin).intersection(set(df.also_bought)))
    all_nodes = list(set(df.asin).intersection(set(df.also_bought)))

    edges = df[['asin', 'also_bought']]

    #Map String ASINs (Product IDs) to Int IDs
    asin_map_dict = pd.Series(edges.asin.append(edges.also_bought).unique()).reset_index(drop=True).to_dict()
    asin_map = {v: k for k, v in asin_map_dict.items()}

    edges.asin = edges.asin.apply(lambda x: asin_map[x])
    edges.also_bought = edges.also_bought.apply(lambda x: asin_map[x])
    edges = edges.reset_index(drop=True)

    #Text Manipulations
    text_df = df[['asin','title','niche']]
    text_df.asin = text_df.asin.apply(lambda x: asin_map[x])
    text_df = text_df.drop_duplicates('asin').reset_index(drop=True)

    #TF-IDF Vectorizer for Title Text Feature
    corpus = list(text_df.title)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    node_features = torch.Tensor(X.toarray())
    node_labels = torch.from_numpy(text_df['niche'].astype('category').cat.codes.to_numpy())
    edges_src = torch.from_numpy(edges['asin'].to_numpy())
    edges_dst = torch.from_numpy(edges['also_bought'].to_numpy())

    return edges, node_features, node_labels, edges_src, edges_dst

def build_graph(node_features, node_labels, edges_src, edges_dst):
    '''
    Builds DGL Graph
    '''
    #Builds graph
    g = dgl.graph((edges_src, edges_dst))
    g.ndata['feat'] = node_features
    g.ndata['label'] = node_labels

    return g

def train_test_split(g):
    '''
    Splits the graph by edges for training and testing

    Source: https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    '''
    # Split edge set for training and testing
    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.number_of_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.number_of_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.number_of_nodes())

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g
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
import dgl.function as fn

class DotPredictor(nn.Module):
    '''
    Dot Predictor of Edges,

    Source:
    https://docs.dgl.ai/en/latest/new-tutorial/4_link_predict.html
    '''
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


#Unzipping Gzip File
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

#To pd DataFrame
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#TF-IDF Vectorizer
def tfidf(corpus):
    '''Given text corpus, returns TF-IDF Vector for each row '''
    corpus = corpus #list(text_df.title)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X.toarray()


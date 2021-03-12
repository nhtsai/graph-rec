from dgl.nn import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    '''
    DGL 2-Layer GraphSAGE Model
    Source: https://docs.dgl.ai/en/0.4.x/_modules/dgl/nn/pytorch/conv/sageconv.html
    '''
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
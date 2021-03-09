# Graph-based Product Recommendation
DSC180B Capstone Project on Graph Data Analysis

Project Website: https://nhtsai.github.io/graph-rec/

## Project
Amazon Product Recommendation using a graph neural network approach.

### Requirements
- dask
- pandas
- torch
- torchtext
- dgl

## Data
Amazon Product Dataset from Professor Julian McAuley ([link](http://jmcauley.ucsd.edu/data/amazon/links.html))
* Product Reviews (5-core)
* Product Metadata
* Product Image Features

## Graph and Features
The graph is a heterogeneous, bipartite user-product graph, connected by reviews.
  * Product Nodes (`ASIN`)
    * Features: `title`, `price`, image representation
  * User Nodes (`reviewerID`)
  * Edges (`user`, `reviewed`, `product`) and (`product`, `reviewed-by`, `user`)
    * Features: `helpful`, `overall`

## Model
The model is an unsupervised PinSage model (adapted from [DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage))

## References
* [GraphSAGE](http://snap.stanford.edu/graphsage/)
* [PinSage](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)

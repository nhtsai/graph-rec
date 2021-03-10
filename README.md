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
### Datasets
Amazon Product Dataset from Professor Julian McAuley ([link](http://jmcauley.ucsd.edu/data/amazon/links.html))
* Product Reviews (5-core)
* Product Metadata
* Product Image Features

## GraphSAGE Model

## PinSAGE

### Graph & Features
The graph is a heterogeneous, bipartite user-product graph, connected by reviews.
 * Product Nodes (`ASIN`)
   * Features: `title`, `price`, image representation
 * User Nodes (`reviewerID`)
 * Edges (`user`, `reviewed`, `product`) and (`product`, `reviewed-by`, `user`)
   * Features: `helpful`, `overall`

### Data Configuration (`config/data-params.json`)

### Model
We use an unsupervised PinSage model (adapted from [DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage)).

### Model Configuration (`config/pinsage-model-params.json`)
- `name`: model configuration name
- `random-walk-length`: maximum number traversals for a single random walk, `default: 2`
- `random-walk-restart-prob`: termination probability after each random walk traversal, `default: 0.5`
- `num-random-walks`:  number of random walks to try for each given node, `default: 10`
- `num-neighbors`: number of neighbors to select for each given node, `default: 3`
- `num-layers`: number of sampling layers, `default: 2`
- `hidden-dims`: dimension of product embedding, `default: 64 or 128`
- `batch-size`: batch size, `default: 64`
- `num-epochs`: number of training epochs, `default: 500`
- `batches-per-epoch`: number of batches per training epoch, `default: 512`
- `num-workers`: number of workers, `default: 3 or (#cores - 1)
- `lr`: learning rate, `default: 3e-4`
- `k`: number of recommendations, `default: 500`
- `model-dir`: directory of existing model to continue training
- `existing-model`: filename of existing model to continue training, `default: null`
- `id-as-features`: use id as features, makes model transductive
- `eval-freq`: evaluates model on validation set when `epoch % eval-freq == 0`, also evaluates model after last training epoch
- `save-freq`: saves model when `epoch % save-freq == 0`, also saves model after last training epoch

## References
* [GraphSAGE Homepage](http://snap.stanford.edu/graphsage/)
* [GraphSAGE Research Paper](https://arxiv.org/abs/1706.02216)
* [PinSage Article](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)
* [PinSage Research Paper](https://arxiv.org/abs/1806.01973)

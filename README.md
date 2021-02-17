# graph-rec
DSC180B Capstone Project on Graph Data Analysis

Project Website: https://nhtsai.github.io/graph-rec/

## Project
Amazon Product Recommendation using a graph-based neural network approach.

### Requirements
- dask
- pandas
- torch
- torchtext
- dgl

## Data
Amazon Product Dataset from Professor Julian McAuley ([link](http://jmcauley.ucsd.edu/data/amazon/links.html))
* Books Product Metadata ([link](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Books.json.gz), 788M)
* Books Product Image Features ([link](http://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Books.b), 35G)

## Model
* Image features (product image)
* Product/Text features (title, price, category, brand, etc.)
* Graph features
  * Nodes: products
  * Edges: also bought, also viewed, bought together

## References
* [GraphSAGE](http://snap.stanford.edu/graphsage/)
* [PinSage](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)
* [Graph based recommendation engine](https://towardsdatascience.com/graph-based-recommendation-engine-for-amazon-products-1a373e639263)

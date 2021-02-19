## Graph-Based Product Recommendation
Project completed by Nathan Tsai and Abdullatif Jarkas as part of DSC 180B Graph Data Analysis course.

## Abstract
Recommender systems are important, revenue-generating technologies in many of the services today, providing recommendations for social, product, and other networks. However, the majority of existing recommender system methods use metrics of similarity to recommend other nodes through content-based and collaborative filtering approaches, which do not take into account the graph structure of the relationships between the nodes. A graph-based recommender system then is able to utilize graph relationships to improve node embeddings for recommendation in a way that conventional recommender systems cannot. Inspired by PinSage<sup>[1]</sup>, we explore an unsupervised graph-based recommendation method that can take advantage of the relationships between nodes, in addition to the text and image features, and generate more accurate and robust embeddings for Amazon product recommendation.

## Introduction
Recommender systems are responsible for large revenues and consumer satisfaction in many of the services used today. Widely-used services, such as Netflix, Facebook, Amazon, and LinkedIn, use recommender systems to suggest movies, posts, users, and products to their consumers. Traditional recommender system methods use metrics of similarity to recommend other products through content-based and collaborative filtering approaches. However, product data can be expressed in a non-Euclidean graph format with relationships, such as products bought together or products viewed together. These recommender system methods do not take into account the graph relationships between the product nodes to improve recommendations. A graph-based approach to recommendation is able to fully utilize the relationships between product nodes, in addition to any product text and image features, to generate more accurate and robust embeddings, compared to embeddings from traditional recommender systems.

### Related Work
Our work builds upon the existing advancement in applying graph neural networks to  recommender systems. Graph convolutional networks (GCNs)<sup>[2]</sup> have allowed deep learning to harness the power of non-Euclidean data, providing relationship and structure data to deep learning techniques. GraphSAGE<sup>[3]</sup> introduced an inductive approach to generating embeddings that sampled neighboring nodes and aggregated their features to produce embeddings. PinSage<sup>[1]</sup> improved upon the GraphSAGE algorithm by introducing a graph-based recommender system with a new sampling and aggregation process and providing an efficient technique for large, web-scale training for production models. We adapt the PinSage<sup>[1]</sup> algorithm to work in an unsupervised learning context to generate more robust and accurate product embeddings that take into account the underlying product graph structure.

### Existing Approaches
- Content-Based Recommendation
- Collaborative Filtering

### Problem

## Methods

### Data

### Model

## Results

## Conclusions

## References
1. R. Ying, R. He, K. Chen, P. Eksombatchai, W. L. Hamilton, J. Leskovec. 2018. Graph Convolutional Neural Networks for Web-Scale Recommender Systems.
2. T. N. Kipf and M. Welling. 2017. Semi-supervised Classification with Graph Convolutional Networks. 
3. W. L. Hamilton, R. Ying, and J. Leskovec. 2017. Inductive Representation Learning on Large Graphs.
4. J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. 2015. Image-based Recommendations on Styles and Substitutes.
5. T. Mikolov, I Sutskever, K. Chen, G. S. Corrado, and J. Dean. 2013. Distributed Representations of Words and Phrases and Their Compositionality.
6. A. Andoni and P. Indyk. 2006. Near-optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions.

[1]: https://arxiv.org/abs/1806.01973 (PinSage)
[2]: https://arxiv.org/abs/1609.02907 (GCN)
[3]: https://arxiv.org/abs/1706.02216 (GraphSAGE)
[4]: https://arxiv.org/abs/1506.04757 (Image Embeddings)
[5]: https://arxiv.org/abs/1310.4546 (Word2Vec)
[6]: https://www.mit.edu/~andoni/papers/cSquared.pdf (LSH)

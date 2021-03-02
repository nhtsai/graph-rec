import pickle
import argparse
import numpy as np
import torch
import dgl


class LatestNNRecommender(object):
    """
    LatestNNRecommender class uses given item embeddings to
    recommend k-nearest neighboring items.
    """

    def __init__(self, user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size):
        """Constructor of LatestNNRecommender class.

        user_ntype (str): user node name
        item_ntype (str): item node name
        user_to_item_etype (str): user-item edge name
        timestamp (str): timestamp column name
        batch_size (int): batch size
        """
        self.user_ntype = user_ntype
        self.item_ntype = item_ntype
        self.user_to_item_etype = user_to_item_etype
        self.batch_size = batch_size
        self.timestamp = timestamp

    def recommend(self, full_graph, K, h_user, h_item):
        """

        Args:
            full_graph (dgl.DGLGraph): bipartite user-item graph
            K (int): number of items to recommend.
            h_user (None): user node embeddings?
            h_item (torch.FloatTensor): item node embeddings

        Returns:
            Returns an (n_user, K) matrix of recommended items for each user.
        """
        # get subgraph of all user-item edges
        graph_slice = full_graph.edge_type_subgraph([self.user_to_item_etype])

        # get latest interaction for each user
        latest_interactions = dgl.sampling.select_topk(
            graph_slice,    # graph
            1,              # number of edges
            self.timestamp, # edge weight
            edge_dir='out'
        )

        # get all users and their latest items
        user, latest_items = latest_interactions.all_edges(form='uv', order='srcdst')

        # each user should have at least one "latest" interaction
        n_users = full_graph.number_of_nodes(self.user_ntype)
        assert torch.equal(user, torch.arange(n_users))

        recommended_batches = []
        user_batches = torch.arange(n_users).split(self.batch_size)
        for user_batch in user_batches:
            latest_item_batch = latest_items[user_batch].to(device=h_item.device)
            dist = h_item[latest_item_batch] @ h_item.t()
            # exclude items that are already interacted
            for i, u in enumerate(user_batch.tolist()):
                interacted_items = full_graph.successors(u, etype=self.user_to_item_etype)
                dist[i, interacted_items] = -np.inf
            recommended_batches.append(dist.topk(K, 1)[1])

        recommendations = torch.cat(recommended_batches, 0)
        return recommendations


def prec(recommendations, ground_truth):
    """Returns hitrate metric @ k of recommendations compared to a ground truth matrix.

    Args:
        recommendations (torch.Tensor): list of recommendations
        ground_truth (scipy.sparse.csr_matrix): validation or test matrix
    """
    n_users, n_items = ground_truth.shape
    K = recommendations.shape[1]
    user_idx = np.repeat(np.arange(n_users), K)
    item_idx = recommendations.flatten()
    relevance = ground_truth[user_idx, item_idx].reshape((n_users, K))
    hit = relevance.any(axis=1).mean()
    return hit


def evaluate_nn(dataset, h_item, k, batch_size, test_mode=False):
    """Evaluates and returns hit-rate @ k metric using the LatestNNRecommender class.

    Args:
        dataset (dict): dictionary of preprocessed dataset features and information
        h_item (torch.FloatTensor): item node embeddings
        k (int): number of neighbors to recommend
        batch_size (int): batch size
        test_mode (bool): evaluate test set
    """
    g = dataset['train-graph']
    # item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    rec_engine = LatestNNRecommender(
        user_ntype, item_ntype, user_to_item_etype, timestamp, batch_size)

    # get k recommendations using embeddings
    recommendations = rec_engine.recommend(g, k, None, h_item).cpu().numpy()

    if test_mode:
        return prec(recommendations, dataset['test-matrix'].tocsr())
    return prec(recommendations, dataset['val-matrix'].tocsr())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('item_embedding_path', type=str)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    with open(args.item_embedding_path, 'rb') as f:
        emb = torch.FloatTensor(pickle.load(f))
    print(evaluate_nn(dataset, emb, args.k, args.batch_size))

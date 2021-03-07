import dgl
import numpy as np
import scipy.sparse as ssp
import dask.dataframe as dd


def train_test_split_by_time(df, timestamp, user):
    """Creates train-test splits of dataset by training on past to predict the future.

    This is the train-test split method most of the recommender system papers running on MovieLens
    takes. It essentially follows the intuition of "training on the past and predict the future".
    One can also change the threshold to make validation and test set take larger proportions.

    Args:
        df (pd.DataFrame): dataframe with user id's and timestamps
        timestamp (str): name of column with timestamp data
        user (str): name of column with user id data

    Returns:
        Train, validation, and test indices of edges, represented as NumPy arrays.
    """
    # Create masks for train, validation, and test sets
    df['train_mask'] = np.ones((len(df),), dtype=np.bool) # all true
    df['val_mask'] = np.zeros((len(df),), dtype=np.bool) # all false
    df['test_mask'] = np.zeros((len(df),), dtype=np.bool) # all false

    # Split dataframe into dask dataframe partitions
    df = dd.from_pandas(df, npartitions=10)

    def train_test_split(df):
        """Sorts dataset by timestamp and creates train, validation, and test mask columns.

        Args:
            df (pd.DataFrame): a dataframe with timestamp data

        Returns:
            A dataframe with train, validation, and test mask columns.
        """
        df = df.sort_values([timestamp]) # sort dataframe by timestamp

        # if more than 1 row, move last row from train mask to test mask
        if df.shape[0] > 1:
            df.iloc[-1, -3] = False
            df.iloc[-1, -1] = True

        # if more than 2 rows, move 2nd to last row from train mask to validation mask
        if df.shape[0] > 2:
            df.iloc[-2, -3] = False
            df.iloc[-2, -2] = True
        return df

    df = df.groupby(user, group_keys=False) \
           .apply(train_test_split) \
           .compute(scheduler='processes') \
           .sort_index()

    # print(df[df[user] == df[user].unique()[0]].sort_values(timestamp))

    return df['train_mask'].to_numpy().nonzero()[0], \
           df['val_mask'].to_numpy().nonzero()[0], \
           df['test_mask'].to_numpy().nonzero()[0]


def build_train_graph(g, train_indices, utype, itype, etype, etype_rev):
    """Builds and returns training subgraph.

    Args:
        g (dgl.DGLGraph): bipartite directed graph between users and items
        train_indices (np.ndarray): indices of train dataset
        utype (str): name of user node
        itype (str): name of item node
        etype (str): name of edge from user to item
        etype_rev (str): name of edge from item to user

    Returns:
        A subgraph induced by edges for training, with node and edge features.
    """
    # Create subgraph induced by train edges
    train_g = g.edge_subgraph(
        {etype: train_indices, etype_rev: train_indices},
        preserve_nodes=True
    )

    # Remove the induced node IDs - should be assigned by model instead
    del train_g.nodes[utype].data[dgl.NID]
    del train_g.nodes[itype].data[dgl.NID]

    # Copy node features to subgraph
    for ntype in g.ntypes:
        for col, data in g.nodes[ntype].data.items():
            train_g.nodes[ntype].data[col] = data

    # Copy edge features to subgraph
    for etype in g.etypes:
        for col, data in g.edges[etype].data.items():
            train_g.edges[etype].data[col] = data[train_g.edges[etype].data[dgl.EID]]

    return train_g


def build_val_test_matrix(g, val_indices, test_indices, utype, itype, etype):
    """Builds and returns validation and test matrices.

    Args:
        g (dgl.DGLGraph): bipartite directed graph between users and items
        val_indices (np.ndarray): indices of validation dataset
        test_indices (np.ndarray): indices of test dataset
        utype (str): name of user node
        itype (str): name of item node
        etype (str): name of edge from user to item

    Returns:
        Validation and test adjacency matrices, represented as scipy.sparse coordinate matrices.
    """
    n_users = g.number_of_nodes(utype) # number of users
    n_items = g.number_of_nodes(itype) # number of items

    # get tensors source and destination nodes for validation edges
    val_src, val_dst = g.find_edges(val_indices, etype=etype)

    # get tensors source and destination nodes for test edges
    test_src, test_dst = g.find_edges(test_indices, etype=etype)

    # convert tensors to numpy arrays
    val_src = val_src.numpy()
    val_dst = val_dst.numpy()
    test_src = test_src.numpy()
    test_dst = test_dst.numpy()

    # create sparse validation adjacency matrix
    val_matrix = ssp.coo_matrix(
        (np.ones_like(val_src), (val_src, val_dst)),
        shape=(n_users, n_items)
    )

    # create sparse test adjacency matrix
    test_matrix = ssp.coo_matrix(
        (np.ones_like(test_src), (test_src, test_dst)),
        shape=(n_users, n_items)
    )

    return val_matrix, test_matrix


def linear_normalize(values):
    """Helper function for column-wise min-max linear normalization."""
    return (values - values.min(0, keepdims=True)) / \
        (values.max(0, keepdims=True) - values.min(0, keepdims=True))

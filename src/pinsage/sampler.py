import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader

def compact_and_copy(frontier, seeds):
    """ Compact and copy graph and destination nodes into a dgl.DGLBlock graph.
    Args:
        frontier (dgl.DGLGraph): a graph made from sampled neighbors using PinSAGE Sampler
        seeds (torch.Tensor): seed node IDs used to generate neighbors

    Returns:
        A dgl.DGLBlock bipartite structured frontier graph with destination seed nodes.
    """
    block = dgl.to_block(frontier, seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

class ItemToItemBatchSampler(IterableDataset):
    """Item to Item Batch Sampler class."""
    def __init__(self, g, user_type, item_type, batch_size):
        """Constructor for ItemToItemBatchSampler class.

        Args:
            g (dgl.DGLGraph): graph of training dataset
            user_type (str): user node name
            item_type (str): item node name
            batch_size (int): batch size
        """
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        """
        Iterator that yields batch-sized torch.Tensors of
        head nodes, tail nodes, and negative tail nodes.
        """
        while True:
            # list of head nodes chosen at random
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            # list of tail nodes after performing random walk from head nodes
            tails = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])[0][:, 2]
            # list of negative tail nodes chosen at random
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))

            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

class NeighborSampler(object):
    """Neighbor Sampler class that uses PinSAGE Sampler."""
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        """Constructor for NeighborSampler class.

        Args:
            g (dgl.DGLGraph): graph of training datset
            user_type (str): user node name
            item_type (str): item node name
            random_walk_length (int): the maximum number traversals for a single random walk
            random_walk_restart_prob (int): termination probability after each traversal
            num_random_walks (int): number of random walks to try for each given node
            num_neighbors (int): number of neighbors (or most commonly visited nodes)
                                 to select for each given node
            num_layers (int): number of sampling layers
        """
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]

        # Create a PinSAGESampler for each layer.
        self.samplers = [
            dgl.sampling.PinSAGESampler(
                g, item_type, user_type,
                random_walk_length, random_walk_restart_prob,
                num_random_walks, num_neighbors
            ) for _ in range(num_layers)
        ]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        """Samples and returns a list of blocks of neighboring nodes.

        Args:
            seeds (torch.Tensor): seed node IDs used to generate neighbors
            heads (torch.Tensor): Optional; head node IDs
            tails (torch.Tensor): Optional; tail node IDs
            neg_tails (torch.Tensor): Optional; negative tail node IDs

        Returns:
            A list of dgl.DGLBlock bipartitie graphs of the neighbors
            to seed nodes sampled in all layers.
        """
        blocks = []
        for sampler in self.samplers:

            # PinSAGE sample of seed node neighbors
            frontier = sampler(seeds)

            # if head nodes are provided, remove head-tail and head-neg_tail edges
            if heads is not None:
                # find edge IDs of head-tail and head-neg_tail edges
                eids = frontier.edge_ids(
                    torch.cat([heads, heads]), torch.cat([tails, neg_tails]),
                    return_uv=True
                )[2]
                if len(eids) > 0:
                    old_frontier = frontier
                    frontier = dgl.remove_edges(old_frontier, eids)
                    #print(old_frontier)
                    #print(frontier)
                    #print(frontier.edata['weights'])
                    #frontier.edata['weights'] = old_frontier.edata['weights'][frontier.edata[dgl.EID]]

            # Create dgl.DGLBlock object
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        """Create a graphs with positive connections and negative connections.

        Args:
            heads (torch.Tensor): head node IDs
            tails (torch.Tensor): tail node IDs after random walk
            neg_tails (torch.Tensor): negative tail node IDs

        Returns:
            A positive-connected graph, a negative connected graph,
            and blocks of sampled neighbors using positive graph seed nodes.
        """
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))

        # find and remove the common isolated nodes across both graphs
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        # get seed node IDs from positive graph
        seeds = pos_graph.ndata[dgl.NID]

        # create sampled neighbor block graphs
        blocks = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks

    # def get_block(self, seeds, ntype, textset=None, imgset=None):
    #     """TODO: docstring"""
    #     blocks = []
    #     for sampler in self.samplers:
    #         frontier = sampler(seeds)
    #         block = compact_and_copy(frontier, seeds)
    #         seeds = block.srcdata[dgl.base.NID]
    #         blocks.insert(0, block)
    #     assign_features_to_blocks(blocks, self.g, textset, imgset, ntype)
    #     return blocks

def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    """Copies data to the given block from the corresponding nodes in the original graph.

    Args:
        ndata (dict[str, Tensor]): node data from dgl.DGLBlock sampled neighbor graph
        g (dgl.DGLGraph): bipartite user-item graph
        ntype (str): node name
        assign_id (bool): Optional; whether to assign node ID
    """
    # induced nodes form the block subgraph
    induced_nodes = ndata[dgl.NID].numpy()

    # for each simple node feature
    for col in g.nodes[ntype].data.keys():
        # skip feature if no asigning id and feature column is node ID
        if not assign_id and col == dgl.NID:
            continue
        # assign block node data to the induced node features
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

def assign_textual_node_features(ndata, textset):
    """
    Assigns numericalized tokens from a torchtext dataset to given block.

    The numericalized tokens would be stored in the block as node features
    with the same name as ``field_name``.

    The length would be stored as another node feature with name
    ``field_name + '__len'``.

    Args:
        block (DGLHeteroGraph):
            First element of the compacted blocks, with "dgl.NID" as the
            corresponding node ID in the original graph, hence the index to the
            text dataset.

            The numericalized tokens (and lengths if available) would be stored
            onto the blocks as new node features.
        textset (torchtext.data.Dataset): A torchtext dataset whose number
            of examples is the same as that of nodes in the original graph.
    """
    # get induced node ids
    node_ids = ndata[dgl.NID].numpy()

    for field_name, field in textset.fields.items():
        # get the text field of each node's textset
        examples = [getattr(textset[i], field_name) for i in node_ids]

        # get tokenized text and lengths
        tokens, lengths = field.process(examples)

        if not field.batch_first:
            tokens = tokens.t()

        ndata[field_name] = tokens
        ndata[field_name + '__len'] = lengths

# def assign_visual_node_features(ndata, imgset):
#     """Assigns image feature from a image dataset dictionary to the given block.

#     Args:
#         ndata (dict[str, torch.Tensor]): node data from dgl.DGLBlock sampled neighbor graph
#         imgset (dict[str, np.ndarray]): image representations from image feature dictionary
#     """
#     node_ids = ndata[dgl.NID].numpy()
#     ndata['image'] = torch.FloatTensor([imgset[i] for i in node_ids])

def assign_features_to_blocks(blocks, g, textset, ntype):
    """For the first block (which is closest to the input),
    copy the features from the original graph as well as the texts.

    Args:
        blocks (list[dgl.DGLBlocks]): sampled neighbor blocks
        g (dgl.DGLGraph): bipartite user-item graph
        textset (torchtext.data.Dataset): text features
        imgset (dict): image features
        ntype (str): item node name
    """
    # assign features to the first block
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_textual_node_features(blocks[0].srcdata, textset)
    # assign_visual_node_features(blocks[0].srcdata, imgset)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)
    assign_textual_node_features(blocks[-1].dstdata, textset)
    # assign_visual_node_features(blocks[-1].dstdata, imgset)

class PinSAGECollator(object):
    """PinSAGECollator class."""
    def __init__(self, sampler, g, ntype, textset):
        """Constructor for PinSAGECollator class.

        Args:
            sampler (NeighborSampler): node neighbor sampler object
            g (dgl.DGLGraph): bipartite user-item graph
            ntype (str): item node name
            textset (torchtext.data.Dataset): text features
            imgset (dict): image features
        """
        self.sampler = sampler
        self.ntype = ntype
        self.g = g
        self.textset = textset
        # self.imgset = imgset

    def collate_train(self, batches):
        """Collates training input samples into batches.

        Args:
            batches (tuple[torch.Tensors]):
                head, tail, and negative tail node IDs
                generated from ItemToItemBatchSampler.

        Returns:
            A dgl.DGLGraph made from positive edges,
            a dgl.DGLGraph made from negative edges, and
            a list of dgl.DGLBlock graphs connecting sampled neighbors to seed nodes.
        """
        heads, tails, neg_tails = batches[0]
        # Construct multilayer neighborhood via PinSAGE...
        pos_graph, neg_graph, blocks = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)

        return pos_graph, neg_graph, blocks

    def collate_test(self, samples):
        """Collates test dataset into blocks

        Args:
            samples (list[int]): a list of sampled node IDs

        Returns:
            A list of dgl.DGLBlock graphs created from
            given sampled node IDs, with item, text, and image features.
        """
        # get test seeds
        batch = torch.LongTensor(samples)
        blocks = self.sampler.sample_blocks(batch)
        assign_features_to_blocks(blocks, self.g, self.textset, self.ntype)
        return blocks

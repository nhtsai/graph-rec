"""Main Pinsage model and train/test code.
Assumes data preprocessing file has been run beforehand.
"""

# built-in imports
import pickle
import os
import json

# third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from tqdm import tqdm

# local imports
import layers
import sampler as sampler_module
import evaluation


class PinSAGEModel(nn.Module):
    """Wrapper class for the PinSAGE model."""

    def __init__(self, full_graph, ntype, textset, hidden_dims, n_layers):
        """Constructor of PinSAGE model class.

        Args:
            full_graph (dgl.DGLGraph): bipartite user-item graph
            ntype (str): item node name
            textset (torchtext.data.Dataset): text features
            hidden_dims (int): dimension of hidden layer
            n_layers (int): number of hidden layers
        """
        super().__init__()

        self.proj = layers.LinearProjector(
            full_graph, ntype, textset, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        """Forward propagation of PinSAGE model.

        Args:
            pos_graph (dgl.DGLGraph): positive-connected graph
            neg_graph (dgl.DGLGraph): negative-connected graph
            blocks (list[dgl.DGLBlock]): sample neighborhood graphs for each layer

        Returns:
           A torch.Tensor of the nonnegative values of (negative scores - positive scores + 1).
        """
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        """Returns the embedded representation given block made from sampling neighboring nodes."""
        # project features
        h_item = self.proj(blocks[0].srcdata)
        # node's own learnable embedding
        h_item_dst = self.proj(blocks[-1].dstdata)

        # embedding + GNN output
        return h_item_dst + self.sage(blocks, h_item)


def train(dataset, model_cfg):
    """Creates and trains a PinSAGE model using Adam optimizer.

    Args:
        dataset (dict[str,*]): dictionary of preprocessed dataset features and information
        model_cfg (dict[str,str]): model configuration parameters

    Returns:
        A trained PinSAGE model.
    """
    g = dataset['train-graph']  # training graph
    # compressed sparse row validation matrix
    val_matrix = dataset['val-matrix'].tocsr()
    # compressed sparse row test matrix
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']  # item text features
    # imgset = dataset['item-images']  # item image features
    user_ntype = dataset['user-type']  # user node
    item_ntype = dataset['item-type']  # item node
    # user-item directed edge
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']  # timestamp column

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # device = torch.device(model_cfg['device'])

    if device.type == 'cpu':
        print('Using CPU...')
    else:
        print("Using CUDA: {}...".format(torch.cuda.current_device()))

    # Assign user and item IDs and use them as features
    # to learn an individual trainable embedding for each entity
    # Note: using ID's as learnable features makes the model transductive,
    #       remove to make model inductive
    if model_cfg['id-as-features']:
        g.nodes[user_ntype].data['id'] = torch.arange(
            g.number_of_nodes(user_ntype))
        g.nodes[item_ntype].data['id'] = torch.arange(
            g.number_of_nodes(item_ntype))

    # Text Features

    for k in item_texts:
        item_texts[k] = item_texts[k].astype(str)
    # Prepare torchtext dataset and vocabulary
    if item_texts is not None:
        fields = {}
        examples = []
        for key, texts in item_texts.items():
            fields[key] = torchtext.data.Field(
                include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)

        textset = torchtext.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            # field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d') # use fasttext

    # Image Features
    # g.nodes[item_ntype].data['image'] = torch.FloatTensor(item_images['image'])

    # Batch Sampler
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, model_cfg['batch-size'])

    # Neighbor Sampler
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype,
        model_cfg['random-walk-length'], model_cfg['random-walk-restart-prob'],
        model_cfg['num-random-walks'], model_cfg['num-neighbors'], model_cfg['num-layers'])

    # Collator
    collator = sampler_module.PinSAGECollator(
        neighbor_sampler, g, item_ntype, textset)

    # Training Data Loader
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=model_cfg['num-workers'])

    # Test Data Loader
    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=model_cfg['batch-size'],
        collate_fn=collator.collate_test,
        num_workers=model_cfg['num-workers'])

    # training Data Loader Iterator
    dataloader_it = iter(dataloader)

    # Model
    model = PinSAGEModel(g, item_ntype, textset,
                         model_cfg['hidden-dims'], model_cfg['num-layers']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=model_cfg['lr'])
    start_epoch = 1

    # load existing model if exists
    if model_cfg['existing-model'] is not None:
        print("Loading existing model: {}...".format(
            model_cfg['existing-model']))
        state = torch.load(
            os.path.join(model_cfg['model-dir'], model_cfg['existing-model']),
            map_location=device
        )
        model.load_state_dict(state['model_state_dict'])
        opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1

    # For each batch of head-tail-negative triplets...
    print("Training model...")
    losses = []
    hits = []
    for epoch_id in range(start_epoch, model_cfg['num-epochs'] + start_epoch):
        batch_losses = []
        # Train
        model.train()
        with tqdm(range(model_cfg['batches-per-epoch'])) as t:
            t.set_description("Training (epoch {})".format(epoch_id))
            for batch_id in t:

                # get next batch of training data
                pos_graph, neg_graph, blocks = next(dataloader_it)

                # Copy to GPU
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)

                # Calculate loss
                loss = model(pos_graph, neg_graph, blocks).mean()

                batch_losses.append(loss.item())
                # Zero optimizer gradients
                opt.zero_grad()

                # Backpropagate loss
                loss.backward()

                # Adjust model weights
                opt.step()

                t.set_postfix(batch=batch_id, loss=loss.item())

        epoch_loss = np.mean(np.array(batch_losses))
        losses.append((epoch_id, epoch_loss))

        # evaluate model on validation set at specified frequency
        if (epoch_id + 1 == model_cfg['num-epochs'] + start_epoch) or \
            epoch_id % model_cfg['eval-freq'] == 0:
            print("Evaluating model...")
            model.eval()
            with torch.no_grad():
                # item batches are groups of node numbers
                item_batches = torch.arange(g.number_of_nodes(
                    item_ntype)).split(model_cfg['batch-size'])
                h_item_batches = []
                # use test dataloader to get sampled neighbors
                for blocks in dataloader_test:
                    # move blocks to GPU
                    for i in range(len(blocks)):
                        blocks[i] = blocks[i].to(device)
                    # get embedding of blocks
                    h_item_batches.append(model.get_repr(blocks))

                # concatenate all embeddings of the batch
                h_item = torch.cat(h_item_batches, 0)  # item node embeddings

                # calculate model evaluation metrics
                hit, precision, recall, _ = evaluation.evaluate(
                    dataset, h_item, model_cfg['k'], model_cfg['batch-size'])

                hits.append((epoch_id, hit))

                # print("Evaluation @ {}: hit: {}, precision: {}, recall: {}".format(model_cfg['k'], hit, precision, recall))
                print("Validation (epoch {}): loss: {:.4f}, hit@{}: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
                    epoch_id, epoch_loss, model_cfg['k'], hit, precision, recall))

        # save model at specified freq or at end of training
        if (epoch_id + 1 == model_cfg['num-epochs'] + start_epoch) or \
                epoch_id % model_cfg['save-freq'] == 0:
            model_dir = "../../data"
            model_fn = "{}_model_{}.pth".format(model_cfg['name'], epoch_id)
            print("Saving model: {}...".format(model_fn))
            state = {
                'epoch': epoch_id,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': epoch_loss,
                'item_embeddings': h_item,
                'k': model_cfg['k'],
                'batch_size': model_cfg['batch-size']
            }
            torch.save(state, os.path.join(model_dir, model_fn))

    return h_item, epoch_id, losses, hits


def test(dataset, model_cfg, item_embeddings, epoch_id=None, use_full_graph=False):
    """Evaluates item embeddings on the test set of interactions.
        Saves item recommendations to a pickle file.

    Args:
        dataset (dict): dataset dictionary
        model_cfg (dict): model configuration
        item_embeddings (torch.Tensor): product embeddings
        use_full_graph (bool): Optional; use full graph for evaluation

    Return:
        Item recommendations in the form of a torch.Tensor of item indices (users, k)
    """
    # evaluate model on test set
    hit, precision, recall, recommendations = evaluation.evaluate(
        dataset, item_embeddings, model_cfg['k'], model_cfg['batch-size'],
        use_test_set=True, use_full_graph=use_full_graph)

    # print evaluation metrics
    print("Test: hit@{}: {:.4f}, precision: {:.4f}, recall: {:.4f}".format(
        model_cfg['k'], hit, precision, recall))

    # save recommendations
    if epoch_id is not None:
        output_fn = model_cfg['name'] + "_results_{}.pkl".format(epoch_id)
    else:
        output_fn = model_cfg['name'] + "_results.pkl"
    with open(os.path.join(model_cfg['model-dir'], output_fn), 'wb') as fp:
        results = {
            "hit": hit,
            "k": model_cfg['k'],
            "precision": precision,
            "recall": recall,
            "recommendations": recommendations
        }
        pickle.dump(results, fp)

    return recommendations


if __name__ == '__main__':
    # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('dataset_path', type=str) # path to dataset pickle file
    # parser.add_argument('--random-walk-length', type=int, default=2)
    # parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    # parser.add_argument('--num-random-walks', type=int, default=10)
    # parser.add_argument('--num-neighbors', type=int, default=3)
    # parser.add_argument('--num-layers', type=int, default=2)
    # parser.add_argument('--hidden-dims', type=int, default=16)
    # parser.add_argument('--batch-size', type=int, default=32)
    # parser.add_argument('--device', type=str, default='cpu') # can also be "cuda:0"
    # parser.add_argument('--num-epochs', type=int, default=1)
    # parser.add_argument('--batches-per-epoch', type=int, default=20000)
    # parser.add_argument('--num-workers', type=int, default=0) # dask workers
    # parser.add_argument('--lr', type=float, default=3e-5)
    # parser.add_argument('-k', type=int, default=10) # number of neighbors to recommend
    # args = parser.parse_args()

    # Load dataset
    data_dir = "../../data"
    dataset_fn = "processed_Amazon_Electronics.pkl"
    with open(os.path.join(data_dir, dataset_fn), 'rb') as f:
        dataset = pickle.load(f)

    # Load config
    config_dir = "../../config"
    config_fn = "pinsage-model-params.json"
    with open(os.path.join(config_dir, config_fn)) as fh:
        model_config = json.load(fh)

    print("Model Config:", model_config)
    item_embeddings, epoch_id, losses, hits = train(dataset, model_config)

    with open(os.path.join(data_dir, "{}_{}_metrics.pkl".format(model_config['name'], epoch_id)), 'wb') as fp:
        pickle.dump({"losses": losses, "hits": hits}, fp)

    test(dataset, model_config, item_embeddings, epoch_id=epoch_id)

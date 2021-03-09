"""
Script that reads from McAuley Amazon Products data and dumps into a pickle
file a heterogeneous graph with categorical and numeric features.
"""

# built-in imports
import os
import gzip
import pickle
import array
import json

# third-party imports
import pandas as pd
import torch

# local imports
# sys.path.insert(0, './src/pinsage')
from builder import PandasGraphBuilder
from data_utils import train_test_split_by_time, build_train_graph, build_val_test_matrix


def parse(path):
    """Generator function ot read .json.gz files."""
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    """Returns a pd.DataFrame of review or metadata from .json.gz filepath."""
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def read_image_features(path):
    """Generator function to read .b image features file."""
    f = open(path, 'rb')
    while True:
        asin = str(f.read(10), 'utf-8')
        if asin == '':
            print('breaking...')
            break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()


def filter_image_features(image_features_path, product_list):
    """Filters image features file based on a given list of ASINs.

    Args:
        image_features_path (str): path to image features file
        product_list (list or np.ndarray): list of product ASINs to keep

    Returns:
        A dictionary of ASIN strings mapped to image feature vectors.
        Each image feature vector consists of 4096 floats.
        Returned dictionary ASINs may be less than in product list if
        missing in image file.
        E.g. {'ASIN': [0.0, ..., 1.0], 'ASIN2': [1.0, ... , 0.9]}
    """
    product_list = set(product_list)
    img_dict = {}
    for d in read_image_features(image_features_path):
        asin, image_vector = d
        if asin in product_list:
            img_dict[asin] = image_vector
    return img_dict


def main(data_cfg):
    """Main processing pipeline for the Amazon Dataset.

    Processes input datasets and saves output pickle files.

    Args:
        data_cfg (dict): dataset configuration parameters
    """
    data_dir = data_cfg['data-dir']
    reviews_fn = data_cfg['reviews-fn']
    metadata_fn = data_cfg['metadata-fn']
    image_fn = data_cfg['image-fn']
    # product_out_fn = data_cfg['product-out-fn']
    # user_out_fn = data_cfg['user-out-fn']
    # image_out_fn = data_cfg['image-out-fn']
    data_out_fn = data_cfg['data-out-fn']

    ### REVIEWS ###
    print("Processing review data...")
    reviews_path = os.path.join(data_dir, reviews_fn)
    reviews = getDF(reviews_path)
    reviews = reviews.drop(['reviewerName', 'reviewTime', 'reviewText'], axis=1).dropna()

    # process reviews features
    reviews['helpful'] = reviews['helpful'].apply(lambda x: x[0]/x[1] if x[1] != 0 else 0.0)

    ### IMAGES ###
    # filter images for products with reviews
    print("Processing image features...")
    image_path = os.path.join(data_dir, image_fn)
    distinct_products_in_reviews = reviews['asin'].unique()
    img_dict = filter_image_features(image_path, distinct_products_in_reviews)

    # save filtered image features dict
    # image_out_path = os.path.join(data_dir, image_out_fn)
    # with open(image_out_path, 'wb') as fp:
    #     print("Saving image dictionary dataset...")
    #     pickle.dump(img_dict, fp)

    distinct_products_all = img_dict.keys()

    ### PRODUCTS ###
    print("Processing product metadata...")
    products_path = os.path.join(data_dir, metadata_fn)
    products = getDF(products_path)

    ### FILTER ###
    print("Filtering reviews and products...")
    # filter reviews for products with image features
    reviews = reviews.copy()[reviews['asin'].isin(distinct_products_all)]

    # filter products with both reviews and images
    products = products.copy()[products['asin'].isin(distinct_products_all)]
    # median impute product prices
    median_price = products['price'].astype(float).median()
    products['price'] = products['price'].astype(float).fillna(median_price)
    products['price'] = (100 * products['price']).astype(int) # convert float price to int feature

    ### USERS ###
    print("Processing users...")
    users = reviews[['reviewerID']].drop_duplicates()

    ### EVENTS ###
    print("Processing interaction events...")
    events = reviews[['reviewerID', 'asin', 'unixReviewTime', 'helpful', 'overall']]

    ### BUILD GRAPH ###
    print("Building graph...")
    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(users, 'reviewerID', 'user')
    graph_builder.add_entities(products, 'asin', 'product')
    graph_builder.add_binary_relations(events, 'reviewerID', 'asin', 'reviewed')
    graph_builder.add_binary_relations(events, 'asin', 'reviewerID', 'reviewed-by')
    g = graph_builder.build()

    ### ADD FEATURES TO GRAPH ###
    print("Adding features to graph...")
    # add product features
    g.nodes['product'].data['price'] = torch.LongTensor(products['price'].values)
    if data_cfg['include-images']:
        g.nodes['product'].data['image'] = \
            torch.FloatTensor([img_dict[i] for i in products['asin'].values])

    # add edge features
    g.edges['reviewed'].data['rating'] = torch.FloatTensor(events['overall'].values)
    g.edges['reviewed'].data['helpful'] = torch.FloatTensor(events['helpful'].values)
    g.edges['reviewed'].data['timestamp'] = torch.LongTensor(events['unixReviewTime'].values)
    g.edges['reviewed-by'].data['rating'] = torch.FloatTensor(events['overall'].values)
    g.edges['reviewed-by'].data['helpful'] = torch.FloatTensor(events['helpful'].values)
    g.edges['reviewed-by'].data['timestamp'] = torch.LongTensor(events['unixReviewTime'].values)

    ### TRAIN, VAL, TEST SPLIT ###
    print("Splitting data...")
    train_indices, val_indices, test_indices = train_test_split_by_time(
        events, 'unixReviewTime', 'reviewerID')

    ### BUILD TRAIN GRAPH ###
    print("Building training graph...")
    train_g = build_train_graph(
        g, train_indices, 'user', 'product', 'reviewed', 'reviewed-by')
    # check for products with no reviews
    assert train_g.out_degrees(etype='reviewed').min() > 0

    ### BUILD VAL & TEST MATRICES ###
    print("Building validation/test matrices...")
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, 'user', 'product', 'reviewed')

    dataset = {
        'full-graph': g,
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': {'title': products['title'].values.astype(str)},
        'item-images': None,
        'user-list': users['reviewerID'].values,
        'product-list': products['asin'].values,
        'image-urls': products['imUrl'].values,
        'user-type': 'user',
        'item-type': 'product',
        'user-to-item-type': 'reviewed',
        'item-to-user-type': 'reviewed-by',
        'timestamp-edge-column': 'timestamp'}

    output_path = os.path.join(data_dir, data_out_fn)
    with open(output_path, 'wb') as f:
        print("Saving processed dataset...")
        pickle.dump(dataset, f)

    return dataset

if __name__ == "__main__":
    print("Reading data config...")
    config_dir = "../../config"
    config_fn = "data-params.json"
    with open(os.path.join(config_dir, config_fn)) as fh:
        data_config = json.load(fh)
    main(data_config)

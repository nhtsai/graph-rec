"""Takes image features file and filters it based on specified asin values."""

import os
import array
import pickle

def read_image_features(path):
    f = open(path, 'rb')
    while True:
        asin = str(f.read(10), 'utf-8')
        if asin == '': 
            print('breaking...')
            break
        a = array.array('f')
        a.fromfile(f, 4096)
        yield asin, a.tolist()

def filter_image_features(data_dir, image_fn, product_list_fn, out_fn):
    image_features_path = os.path.join(data_dir, image_fn)
    product_list_path = os.path.join(data_dir, product_list_fn)
    out_path = os.path.join(data_dir, out_fn)

    # read in product list
    print("Reading product list from {}...".format(product_list_path))
    with open (product_list_path, 'rb') as fp:
        product_list = pickle.load(fp)
    product_list = set(product_list)
    
    print("Filtering image features from {}.".format(image_features_path))
    img_dict = {}
    for d in read_image_features(image_features_path):
        asin, image_vector = d
        if asin in product_list:
            img_dict[asin] = image_vector
            
    print("Dumping dictionary output to {}...".format(out_path))
    with open(out_path, 'wb') as outfp:
        pickle.dump(img_dict, outfp)
    

if __name__ == '__main__':
    # get directory of data files
    data_dir = '../../data'
    image_fn = 'image_features_Electronics.b'
    product_list_fn = 'products.out'
    out_fn = 'image_processed.out'
    filter_image_features(data_dir, image_fn, product_list_fn, out_fn)
    print("finished!")
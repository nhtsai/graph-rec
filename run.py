import sys
import os
import json

from src.pinsage import model
from src.pinsage import process_amazon

def main(targets):

    if 'help' in targets:
        help_msg = """  Possible targets:
    help: program help message
    data: processes data
    pinsage: trains pinsage model on train data
    graphsage: trains graphsage model on train data
    test: runs model on test data
    all (default): trains and tests both models
    clean: removes all output files"""
        print(help_msg)
        return

    if len(targets) == 0 or 'all' in targets:
        targets = ['data', 'pinsage', 'graphsage', 'test']

    # if 'test' in targets:
    #     with open("config/{0}-test-params.json".format(model)) as fh:
    #         data_cfg = json.load(fh)

    # process data using config
    if 'data' in targets:
        config_dir = "./config"
        config_fn = "data-params.json"
        with open(os.path.join(config_dir, config_fn)) as fh:
            data_cfg = json.load(fh)
        dataset = process_amazon.main(data_cfg)

    # train model using config
    pinsage_model, graphsage_model = None, None

    if 'pinsage' in targets:
        # Load config
        config_dir = "./config"
        config_fn = "pinsage-model-params.json"
        with open(os.path.join(config_dir, config_fn)) as fh:
            pinsage_model_cfg = json.load(fh)

        print("Training model embeddings...")
        item_embeddings = model.train(dataset, pinsage_model_cfg)

        if 'test' in targets:
            print("Testing model embeddings...")
            rec = model.test(dataset, pinsage_model_cfg, item_embeddings)


    if 'graphsage' in targets:
        config_dir = "./config"
        config_fn = "graphsage-model-params.json"
        with open(os.path.join(config_dir, config_fn)) as fh:
            graphsage_model_cfg = json.load(fh)
        graphsage_model = train(data_cfg, graphsage_model_cfg)
        graphsage_model.save() # save model as pth

    return


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)

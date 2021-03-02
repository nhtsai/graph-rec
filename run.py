import sys
# import os
import json

# from src.pinsage import evaluation

def model_build(model_name, model_cfg):
    if model_name == 'pinsage':
        pass

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
        with open("config/data-params.json") as fh:
            data_cfg = json.load(fh)
        data = process_data(data_cfg)
        save(data) # save processed data as pkl

    # train model using config
    pinsage_model, graphsage_model = None, None

    if 'pinsage' in targets:
        with open('config/pinsage-model-params.json') as fh:
            pinsage_model_cfg = json.load(fh)
        pinsage_model = train(data_cfg, pinsage_model_cfg)
        pinsage_model.save() # save model as pth

    if 'graphsage' in targets:
        with open('config/graphsage-model-params.json') as fh:
            graphsage_model_cfg = json.load(fh)
        graphsage_model = train(data_cfg, graphsage_model_cfg)
        graphsage_model.save() # save model as pth

    if 'test' in targets:
        recommendations = evaluate(dataset, args, test_mode=True)


# load model state_dict?
# load in val matrix from pkl
# evaluate on validation
# load in test matrix from pkl
# evaluate on test
# get recommendations for one product

    return


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)

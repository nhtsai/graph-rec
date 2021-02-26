import sys
import os
import json

# sys.path.insert(0, 'src')

def model_build(model_name, model_cfg):
    if model_name == 'pinsage':
        pass

def main(targets):
    
    if 'pinsage' in targets:
        model = 'pinsage'
    else:
        model = 'graphsage'

    if 'test' in targets:
        with open("config/{0}-test-params.json".format(model) as fh:
            data_cfg = json.load(fh)

    if 'data' in targets:
        with open("config/{0}-data-params.json".format(model)) as fh:
            data_cfg = json.load(fh)
        
    # data target
    # data = get_data(**data_cfg)


    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        # model_build(feats, labels, **model_cfg)

    return


if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)

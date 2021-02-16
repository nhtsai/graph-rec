import sys
import os
import json

sys.path.insert(0, 'src')

def main(targets):

    if 'test' in targets:
        with open('config/test-params.json') as fh:
            data_cfg = json.load(fh)
        targets = ['features', 'model']

    if 'data' in targets:
        with open('config/data-params.json') as fh:
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

import argparse
import datasets
import engine
import models
import torch
import sys
sys.path.insert(0, '/home/ruochenl/graph/moma-model/momaapi')
from momaapi import MOMA

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=1, type=int)

parser.add_argument('--data_dir', default='/home/ruochenl/data/moma', type=str)
parser.add_argument('--save_dir', default='/home/ruochenl/graph/moma-model/ckpt', type=str)
parser.add_argument('--model_path', default='/home/ruochenl/graph/moma-model/model.pkl', type=str)
parser.add_argument('--feats_dir', default='/home/ruochenl/graph/moma-model/feats', type=str)

parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--oracle', default=False, type=bool)
parser.add_argument('--test', default=False, type=bool)
parser.add_argument('--sub_activity', default=False, type=bool)

def main():
    cfg = parser.parse_args()

    moma_api = MOMA(cfg.data_dir, load_val=True)
    # print(moma_api.statistics)
    print(cfg.oracle)
    print(cfg.sub_activity)
    print(cfg.test)

    if not cfg.test:
        if cfg.sub_activity:
            dataset_train = datasets.MOMASubActivity(cfg, moma_api, split='train', fetch="pyg")
            dataset_val = datasets.MOMASubActivity(cfg, moma_api, split='val', fetch="pyg")
            model = models.SubActivityModel(cfg)

        else:
            print("activity")
            dataset_train = datasets.MOMAActivity(cfg, moma_api, split='train', fetch="pyg")
            dataset_val = datasets.MOMAActivity(cfg, moma_api, split='val', fetch="pyg")
            model = models.ActivityModel(cfg)

        trainer = engine.Trainer(cfg)
        trainer.fit(model, dataset_train, dataset_val)

    else:
        if cfg.sub_activity:
            dataset_test = datasets.MOMASubActivity(cfg, moma_api, split='test', fetch="pyg")
            model = models.SubActivityModel(cfg)

        else:
            print("activity")
            dataset_test = datasets.MOMAActivity(cfg, moma_api, split='test', fetch="pyg")
            model = models.ActivityModel(cfg)

        model.load_state_dict(torch.load(cfg.model_path))
        trainer = engine.Trainer(cfg)
        trainer.test(model, dataset_test)



if __name__ == '__main__':
  main()
import argparse
import datasets
import engine
import models
import torch
import sys
sys.path.insert(0, '/home/ruochenl/graph/moma-model/momaapi')
from momaapi import MOMA

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int)

parser.add_argument('--data_dir', default='/home/ruochenl/data/moma', type=str)
parser.add_argument('--save_dir', default='/home/ruochenl/graph/moma-model/ckpt', type=str)
parser.add_argument('--model_path', default='/home/ruochenl/graph/moma-model/model_pred_cls.pkl', type=str)
parser.add_argument('--feats_dir', default='/home/ruochenl/graph/moma-model/feats', type=str)

parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.005, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--oracle', default=True, type=bool)
parser.add_argument('--test', default=False, type=bool)

def main():
    cfg = parser.parse_args()

    moma_api = MOMA(cfg.data_dir, load_val=True)

    if not cfg.test:

        dataset_train = datasets.MOMAHoi(cfg, moma_api, split='train', fetch="pyg")
        dataset_val = datasets.MOMAHoi(cfg, moma_api, split='val', fetch="pyg")
        model = models.PredicateCLSModel(cfg)
        trainer = engine.TrainerHOI(cfg)
        trainer.fit(model, dataset_train, dataset_val)

    else:

        dataset_test = datasets.MOMAHoi(cfg, moma_api, split='test', fetch="pyg")
        model = models.PredicateCLSModel(cfg)
        model.load_state_dict(torch.load(cfg.model_path))
        trainer = engine.TrainerHOI(cfg)
        trainer.test(model, dataset_test)

if __name__ == '__main__':
  main()
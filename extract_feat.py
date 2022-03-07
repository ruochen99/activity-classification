"""
Extract actor and object features
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.ops as ops
import torchvision.transforms as transforms

import datasets
import utils
from momaapi import MOMA

torch.multiprocessing.set_sharing_strategy('file_system')

class FeatExtractorModel(nn.Module):
  def __init__(self):
    super(FeatExtractorModel, self).__init__()
    self.net = models.resnet18(pretrained=True)
    self.net.layer4.register_forward_hook(self.hook_fn)
    self.buffer = {}

  def hook_fn(self, module, input, output):
    self.buffer[input[0].device] = output

  def forward(self, video):
    self.net(video)
    return self.buffer[video.device]


class FeatExtractor:
  def __init__(self, cfg):
    self.cfg = cfg
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(self.device)

  @staticmethod
  def extract_bboxes(hoi_anns, oracle=True):
    bboxes = []
    if oracle:
      for i, hoi in enumerate(hoi_anns):
        for actor in hoi.actors:
          x1, y1 = actor.bbox.x1, actor.bbox.y1
          x2, y2 = actor.bbox.x2, actor.bbox.y2
          bboxes.append([i, x1, y1, x2, y2])
        for object in hoi.objects:
          x1, y1 = object.bbox.x1, object.bbox.y1
          x2, y2 = object.bbox.x2, object.bbox.y2
          bboxes.append([i, x1, y1, x2, y2])

    else:
      for i, hoi in enumerate(hoi_anns):
        actors = hoi[0]
        objects = hoi[1]
        for j in len(actors['bbox']):
          x1 = actors['bbox'][j][0]
          y1 = actors['bbox'][j][1]
          x2 = x1 + actors['bbox'][j][2]
          y2 = y1 + actors['bbox'][j][3]
          bboxes.append([i, x1, y1, x2, y2])
        for k in len(objects['bbox']):
          x1 = objects['bbox'][k][0]
          y1 = objects['bbox'][k][1]
          x2 = x1 + objects['bbox'][k][2]
          y2 = y1 + objects['bbox'][k][3]
          bboxes.append([i, x1, y1, x2, y2])

    return bboxes


  def fit(self, model, dataset, oracle=True):
    feat_list, sact_ids = [], []

    model = model.to(self.device)
    model = nn.DataParallel(model)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=self.cfg.num_workers)

    model.eval()
    with torch.no_grad():
      for i, (sact_id, hoi_anns, video) in enumerate(dataloader):
        print('[{}] {}: len={}'.format(i, sact_id, video.shape[0]))
        assert video.shape[0] == len(hoi_anns)

        bboxes = self.extract_bboxes(hoi_anns, oracle)
        bboxes = torch.Tensor(bboxes).to(self.device)
        video = transform(video)
        video = video.to(self.device)

        # save memory
        if video.shape[0] > self.cfg.batch_size:
          print('split')
          num_steps = math.ceil(video.shape[0]/self.cfg.batch_size)
          feat = []
          for step in range(num_steps):
            start = step*self.cfg.batch_size
            end = min((step+1)*self.cfg.batch_size, video.shape[0])
            ret = model(video[start:end])
            feat.append(ret)
          feat = torch.cat(feat, dim=0)
        else:
          feat = model(video)

        try:
          feat = ops.roi_align(feat, bboxes, (7, 7), 1/32)
          feat = F.adaptive_avg_pool2d(feat, (1, 1))
          feat = torch.flatten(feat, 1)
          assert feat.shape[0] == bboxes.shape[0]
          feat_list.append(feat.detach().cpu())
          sact_ids.append(sact_id)
        except:
          print(feat.shape)
          print(bboxes.shape)


    feats, chunk_sizes = utils.cat_vl(feat_list)
    # print(chunk_sizes)
    # print(feats.shape)

    os.makedirs(self.cfg.feats_dir, exist_ok=True)
    torch.save(feats, os.path.join(self.cfg.feats_dir, 'feats.pt'))
    torch.save(chunk_sizes, os.path.join(self.cfg.feats_dir, 'chunk_sizes.pt'))
    with open(os.path.join(self.cfg.feats_dir, 'sact_ids.txt'), 'w+') as f:
      f.write('\n'.join(sact_ids))


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/home/ruochenl/data/moma', type=str)
parser.add_argument('--feats_dir', default='/home/ruochenl/graph/moma-model/feats', type=str)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--oracle', default=True, type=bool)
parser.add_argument('--split', default=None, type=str)
# parser.add_argument('--split_by', default='untrim', type=str, choices=['trim', 'untrim'])


def main():
  cfg = parser.parse_args()
  model = FeatExtractorModel()
  moma_api = MOMA(cfg.data_dir, load_val=True)
  dataset = datasets.MOMASubActivity(cfg, moma_api, split=cfg.split, fetch='video')
  feat_extractor = FeatExtractor(cfg)
  feat_extractor.fit(model, dataset)


if __name__ == '__main__':
  main()

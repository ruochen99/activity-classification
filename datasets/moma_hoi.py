import os
import torch
from torchvision import datasets, io
import torchvision.transforms.functional as F

import utils

PRED_ACT_DIR = "/home/alanzluo/data/moma/detection/actor"
PRED_OBJ_DIR = "/home/alanzluo/data/moma/detection/object"


class MOMAHoi(datasets.VisionDataset):
    def __init__(self, cfg, moma_api, split=None, fetch=None):
        super(MOMAHoi, self).__init__(cfg.data_dir)
        self.cfg = cfg
        self.fetch = fetch
        self.moma_api = moma_api

        sact_ids = self.moma_api.get_ids_sact(split)  # ['00000'...]
        self.sact_ids = []
        for sact_id in sact_ids:
            hoi_ids = self.moma_api.get_ids_hoi(ids_sact=[sact_id])
            hoi_anns = self.moma_api.get_anns_hoi(hoi_ids)
            total_num_nodes = min([hoi_ann.num_nodes for hoi_ann in hoi_anns])
            if total_num_nodes > 0:
                self.sact_ids.append(sact_id)
        self.hoi_ids = self.moma_api.get_ids_hoi(ids_sact=self.sact_ids)
        self.feats_map = self.load_feats()

    def load_feats(self, feats_fname='feats.pt', chunk_sizes_fname='chunk_sizes.pt', sact_ids_fname='sact_ids.txt'):
        all_feats = torch.load(os.path.join(self.cfg.feats_dir, feats_fname))
        node_video_chunk_sizes = torch.load(os.path.join(self.cfg.feats_dir, chunk_sizes_fname))  # split nodes by video
        with open(os.path.join(self.cfg.feats_dir, sact_ids_fname), 'r') as f:
            all_sact_ids = f.read().splitlines()
        all_feats = utils.split_vl(all_feats, node_video_chunk_sizes)

        indices = [all_sact_ids.index(sact_id) for sact_id in self.sact_ids]
        feats = [all_feats[index] for index in indices]

        feats_map = {}
        for i in range(len(self.sact_ids)):
            feat = feats[i]
            sact_id = self.sact_ids[i]
            hoi_ids = self.moma_api.get_ids_hoi(ids_sact=[sact_id])
            hoi_anns = self.moma_api.get_anns_hoi(hoi_ids)
            node_frame_chunk_sizes = [hoi_ann.num_nodes for hoi_ann in hoi_anns]
            feat_list = utils.split_vl(feat, node_frame_chunk_sizes)
            for hoi_ann, f in zip(hoi_anns, feat_list):
                feats_map[hoi_ann.id] = f

        return feats_map

    def __getitem__(self, index):
        hoi_id = self.hoi_ids[index]
        hoi_ann = self.moma_api.get_anns_hoi(ids_hoi=[hoi_id])[0]
        return utils.to_pyg_data_hoi(hoi_ann, self.cfg.oracle, self.feats_map[hoi_id])

    def __len__(self):
        return len(self.hoi_ids)




import os
import torch
from torchvision import datasets, io
import torchvision.transforms.functional as F

import utils

NUM_SACT_CLASSES = 97
PRED_ACT_DIR = "/home/alanzluo/data/moma/detection/actor"
PRED_OBJ_DIR = "/home/alanzluo/data/moma/detection/object"


class MOMASubActivity(datasets.VisionDataset):
    def __init__(self, cfg, moma_api, split=None, fetch=None):
        super(MOMASubActivity, self).__init__(cfg.data_dir)
        self.cfg = cfg
        self.fetch = fetch
        self.moma_api = moma_api
        sact_ids = self.moma_api.get_ids_sact(split) # ['00000'...]
        self.sact_ids = []
        if cfg.oracle:
            for sact_id in sact_ids:
                hoi_ids = self.moma_api.get_ids_hoi(ids_sact=[sact_id])
                hoi_anns = self.moma_api.get_anns_hoi(hoi_ids)
                total_num_nodes = min([hoi_ann.num_nodes for hoi_ann in hoi_anns])
                if total_num_nodes > 0:
                    self.sact_ids.append(sact_id)
        else:
            print("non oracle")
            for sact_id in sact_ids:
                non_empty = True
                for id_hoi in self.moma_api.get_ids_hoi(ids_sact=[sact_id]):
                    actors = torch.load(os.path.join(PRED_ACT_DIR, id_hoi))
                    objects = torch.load(os.path.join(PRED_OBJ_DIR, id_hoi))
                    if len(actors['bbox']) == 0 and len(objects['bbox']) == 0:
                        non_empty = False
                if non_empty:
                    self.sact_ids.append(sact_id)
        print(len(self.sact_ids))
        self.add_cfg()
        if self.fetch == 'pyg':
            self.feats = self.load_feats()

    def add_cfg(self):
        setattr(self.cfg, 'num_sact_classes', NUM_SACT_CLASSES)

    def load_feats(self, feats_fname='feats.pt', chunk_sizes_fname='chunk_sizes.pt', sact_ids_fname='sact_ids.txt'):
        all_feats = torch.load(os.path.join(self.cfg.feats_dir, feats_fname))
        node_video_chunk_sizes = torch.load(os.path.join(self.cfg.feats_dir, chunk_sizes_fname))  # split nodes by video
        with open(os.path.join(self.cfg.feats_dir, sact_ids_fname), 'r') as f:
            all_sact_ids = f.read().splitlines()
        all_feats = utils.split_vl(all_feats, node_video_chunk_sizes)

        indices = [all_sact_ids.index(sact_id) for sact_id in self.sact_ids]
        feats = [all_feats[index] for index in indices]

        return feats

    def __getitem__(self, index):
        sact_id = self.sact_ids[index]
        hoi_ids = self.moma_api.get_ids_hoi(ids_sact=[sact_id])
        sact = self.moma_api.get_anns_sact([sact_id])[0]
        if self.cfg.oracle:
            hoi_anns = self.moma_api.get_anns_hoi(hoi_ids)
            if self.fetch == 'video':
                video_images = []
                for hoi_id in hoi_ids:
                    image_path = self.moma_api.get_paths(ids_hoi=[hoi_id])[0]
                    image = io.read_image(image_path).float()
                    video_images.append(image)
                videos = torch.stack(video_images)
                return sact_id, hoi_anns, videos

            elif self.fetch == 'pyg':
                return utils.to_pyg_data(hoi_anns, sact.cid, sact.id, self.cfg.oracle, self.feats[index])
        else:
            hoi_anns = {}
            for hoi_id in hoi_ids:
                hoi_anns["actors"] = torch.load(os.path.join(PRED_ACT_DIR, hoi_id))
                hoi_anns["objects"] = torch.load(os.path.join(PRED_OBJ_DIR, hoi_id))
            if self.fetch == 'video':
                video_images = []
                for hoi_id in hoi_ids:
                    image_path = self.moma_api.get_paths(ids_hoi=[hoi_id])[0]
                    image = io.read_image(image_path).float()
                    video_images.append(image)
                videos = torch.stack(video_images)
                return sact_id, hoi_anns, videos
            elif self.fetch == 'pyg':
                return utils.to_pyg_data(hoi_anns, sact.cid, sact.id, oracle=self.cfg.oracle)


    def __len__(self):
        return len(self.sact_ids)




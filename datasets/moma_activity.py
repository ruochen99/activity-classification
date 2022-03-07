import os
import torch
from torchvision import datasets
import utils

NUM_ACT_CLASSES = 20
PRED_ACT_DIR = "/home/alanzluo/data/moma/detection/actor"
PRED_OBJ_DIR = "/home/alanzluo/data/moma/detection/object"

class MOMAActivity(datasets.VisionDataset):
    def __init__(self, cfg, moma_api, split=None, fetch=None):
        super(MOMAActivity, self).__init__(cfg.data_dir)
        self.cfg = cfg
        self.fetch = fetch
        self.moma_api = moma_api
        act_ids = self.moma_api.get_ids_act(split) # ['00000'...]

        self.act_ids = []
        self.sact_ids = []
        if cfg.oracle:
            for act_id in act_ids:
                sact_ids = self.moma_api.get_ids_sact(ids_act=[act_id])
                tmp_sact_ids = []
                for sact_id in sact_ids:
                    hoi_ids = self.moma_api.get_ids_hoi(ids_sact=[sact_id])
                    hoi_anns = self.moma_api.get_anns_hoi(hoi_ids)
                    total_num_nodes = min([hoi_ann.num_nodes for hoi_ann in hoi_anns])
                    if total_num_nodes > 0:
                        self.sact_ids.append(sact_id)
                        tmp_sact_ids.append(sact_id)
                if len(tmp_sact_ids) > 0:
                    self.act_ids.append(act_id)
        else:
            print("non oracle")
            for act_id in act_ids:
                non_empty = True
                for id_hoi in self.moma_api.get_ids_hoi(ids_act=[act_id]):
                    actors = torch.load(os.path.join(PRED_ACT_DIR, id_hoi))
                    objects = torch.load(os.path.join(PRED_OBJ_DIR, id_hoi))
                    if len(actors['bbox']) == 0 and len(objects['bbox']) == 0:
                        non_empty = False
                if non_empty:
                    self.act_ids.append(act_id)
        print(len(self.act_ids))
        self.add_cfg()
        if self.fetch == 'pyg':
            self.feats = self.load_feats()

    def add_cfg(self):
        setattr(self.cfg, 'num_act_classes', NUM_ACT_CLASSES)

    def load_feats(self, feats_fname='feats.pt', chunk_sizes_fname='chunk_sizes.pt', sact_ids_fname='sact_ids.txt'):
        all_feats = torch.load(os.path.join(self.cfg.feats_dir, feats_fname))
        node_video_chunk_sizes = torch.load(os.path.join(self.cfg.feats_dir, chunk_sizes_fname))  # split nodes by video
        with open(os.path.join(self.cfg.feats_dir, sact_ids_fname), 'r') as f:
            all_sact_ids = f.read().splitlines()
        all_feats = utils.split_vl(all_feats, node_video_chunk_sizes)

        # indices = [all_sact_ids.index(sact_id) for sact_id in self.sact_ids]
        feats = {}
        for sact_id in self.sact_ids:
            feats[sact_id] = all_feats[all_sact_ids.index(sact_id)]
        return feats

    def __getitem__(self, index):
        act_id = self.act_ids[index]
        sact_ids = self.moma_api.get_ids_sact(ids_act=[act_id])
        sact_ids = [sact_id for sact_id in sact_ids if sact_id in self.sact_ids]

        if self.fetch == 'pyg':
            act = self.moma_api.get_ann_act(act_id)
            feats = [self.feats[sact_id] for sact_id in sact_ids]
            return utils.to_pyg_data_act(act.cid, act.id, sact_ids, feats, self.moma_api, self.cfg.oracle)

    def __len__(self):
        return len(self.act_ids)




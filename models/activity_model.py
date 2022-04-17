import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .layers import MLP
from .encoders import GINEEncoder, GCNEncoder
from .decoders import ActHead
import utils

DIM_ORC_NODE_ATTR = 26 + 125
# DIM_ORC_EDGE_ATTR = 22 + 4 + 39 + 11
DIM_NODE_ATTR = 512



class ActivityModel(nn.Module):
    def __init__(self, cfg, dim=1024):
        super(ActivityModel, self).__init__()
        self.cfg = cfg
        dim_hidden = dim*2
        self.encoder = GCNEncoder(dim=dim_hidden)
        self.mlp_node = MLP(DIM_NODE_ATTR, dim)
        # if cfg.oracle:
        self.mlp_orc_node = MLP(DIM_ORC_NODE_ATTR, dim)
        self.act_head = ActHead(num_classes=cfg.num_act_classes, dim=dim_hidden)

    def get_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
        return scheduler

    def forward(self, data):
        node_attr = self.mlp_node(data.x)
        if self.cfg.oracle:
            orc_node_attr = self.mlp_orc_node(data.orc_node_attr)
            # orc_edge_attr = self.mlp_orc_edge(data.orc_edge_attr)
            node_attr = torch.cat([node_attr, orc_node_attr], dim=-1)
            # edge_attr = orc_edge_attr
        else:
            pred_node_attr = self.mlp_orc_node(data.pred_node_attr)
            node_attr = torch.cat([node_attr, pred_node_attr], dim=-1)

        # embed = self.encoder(data.edge_index, node_attr, edge_attr)
        embed = self.encoder(data.edge_index, node_attr)
        logits_act = self.act_head(embed, data.batch)
        loss_sact = F.cross_entropy(logits_act, data.act_cid)
        acc_sact = utils.get_acc(logits_act, data.act_cid)
        mAP_sact = utils.get_mAP(logits_act, data.act_cid)
        stats = {
            'loss_act': (loss_sact.item(), logits_act.shape[0]),
            'acc_act': (acc_sact, logits_act.shape[0]),
            'mAP_act': (mAP_sact, logits_act.shape[0]),
        }
        # preds = logits_act.argmax(axis=1)
        return loss_sact, stats, logits_act





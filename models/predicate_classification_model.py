import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .layers import MLP
from .encoders import GINEEncoder, GCNEncoder
from .decoders import SActHead
import utils

DIM_ORC_NODE_ATTR = 26 + 125
# DIM_ORC_EDGE_ATTR = 22 + 4 + 39 + 11
DIM_NODE_ATTR = 512

NUM_REL_CLASSES = 19
NUM_ATT_CLASSES = 4
NUM_TA_CLASSES = 33
NUM_IA_CLASSES = 9

class PredicateCLSHead(nn.Module):
    def __init__(self, dim_in=1024, dim_out=1024):
        super(PredicateCLSHead, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PredicateCLSModel(nn.Module):
    def __init__(self, cfg, dim=512):
        super(PredicateCLSModel, self).__init__()
        self.cfg = cfg
        dim_hidden = dim*2 if cfg.oracle else dim
        self.encoder = GCNEncoder(dim=dim_hidden)
        self.mlp_node = MLP(DIM_NODE_ATTR, dim)
        if cfg.oracle:
            self.mlp_orc_node = MLP(DIM_ORC_NODE_ATTR, dim)
            # self.mlp_orc_edge = MLP(DIM_ORC_EDGE_ATTR, dim_hidden)

        self.att_head = PredicateCLSHead(dim_hidden, NUM_ATT_CLASSES)
        self.ia_head = PredicateCLSHead(dim_hidden, NUM_IA_CLASSES)
        self.rel_head = PredicateCLSHead(dim_hidden*2, NUM_REL_CLASSES)
        self.ta_head = PredicateCLSHead(dim_hidden*2, NUM_TA_CLASSES)

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

        # embed = self.encoder(data.edge_index, node_attr, edge_attr)
        embed = self.encoder(data.edge_index, node_attr)

        att_pred = self.att_head(embed)
        ia_pred = self.ia_head(embed)

        loss = 0
        loss_att = F.binary_cross_entropy_with_logits(att_pred, data.node_att)
        loss_ia = F.binary_cross_entropy_with_logits(ia_pred, data.node_ia)
        loss += loss_att
        loss += loss_ia

        num_edge = len(data.edge_index[0])
        rel_preds = torch.zeros((num_edge, NUM_REL_CLASSES))
        ta_preds = torch.zeros((num_edge, NUM_TA_CLASSES))
        for i in range(num_edge):
            src = data.edge_index[0][i]
            trg = data.edge_index[1][i]
            nodes = torch.cat([embed[src], embed[trg]], dim=-1)
            rel_pred = self.rel_head(nodes)
            ta_pred = self.ta_head(nodes)
            rel_preds[i] = rel_pred
            ta_preds[i] = ta_pred

        loss_rel = F.binary_cross_entropy_with_logits(rel_preds, data.edge_rel.cpu())
        loss_ta = F.binary_cross_entropy_with_logits(ta_preds, data.edge_ta.cpu())

        loss += loss_rel
        loss += loss_ta

        preds = {"att": att_pred, "ia": ia_pred, "rel": rel_preds, "ta": ta_preds}

        return loss, preds


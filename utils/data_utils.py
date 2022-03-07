import numpy as np
import torch
from torch_geometric.data import Batch, Data
import os
from torch.utils.data.dataloader import default_collate

NUM_ACT_CLASSES = 20
NUM_SACT_CLASSES = 91
NUM_ACTOR_CLASSES = 26
NUM_OBJECT_CLASSES = 126
NUM_REL_CLASSES = 19
NUM_ATT_CLASSES = 4
NUM_TA_CLASSES = 33
NUM_IA_CLASSES = 9

PRED_ACT_DIR = "/home/alanzluo/data/moma/detection/actor"
PRED_OBJ_DIR = "/home/alanzluo/data/moma/detection/object"


def cat_vl(tensor_list):
  """ Concatenate tensors of varying lengths
  :param tensor_list: a list of tensors of varying tensor.shape[0] but same tensor.shape[1:]
  :return: a concatenated tensor and chunk sizes
  """
  chunk_sizes = torch.LongTensor([tensor.shape[0] for tensor in tensor_list])
  tensor = torch.cat(tensor_list, dim=0)
  return tensor, chunk_sizes


def split_vl(tensor, chunk_sizes):
  """ Split a tensor into sub-tensors of varying lengths
  """
  if isinstance(chunk_sizes, torch.Tensor) or isinstance(chunk_sizes, np.ndarray):
    chunk_sizes = chunk_sizes.tolist()
  return list(torch.split(tensor, chunk_sizes))


def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        # non_empty_batch = list(filter(lambda elem: len(elem.x) > 0 and len(elem.edge_index) > 0 and len(elem.edge_attr) > 0, batch))
        data = Batch.from_data_list(data_list=batch)
        repeats = [graph.num_nodes for graph in batch]
        batch_vec = [torch.full((n,), i) for i, n in enumerate(repeats)]
        batch_vec = torch.cat(batch_vec, dim=0)
        setattr(data, "batch", batch_vec)
        return data
    raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

# for creating a complete graph
def build_edge_idx(num_nodes):
    # Initialize edge index matrix
    E = torch.zeros((2, num_nodes * (num_nodes - 1)), dtype=torch.long)

    # Populate 1st row
    for node in range(num_nodes):
        for neighbor in range(num_nodes - 1):
            E[0, node * (num_nodes - 1) + neighbor] = node

    # Populate 2nd row
    neighbors = []
    for node in range(num_nodes):
        neighbors.append(list(np.arange(node)) + list(np.arange(node + 1, num_nodes)))
    E[1, :] = torch.Tensor([item for sublist in neighbors for item in sublist])

    return E

def to_pyg_data(hoi_anns, sact_cid, id, oracle=True, feats=None):
    data_list = []
    if oracle:
        node_frame_chunk_sizes = [hoi_ann.num_nodes for hoi_ann in hoi_anns]
    else:
        node_frame_chunk_sizes = [len(hoi_ann['actors']['bbox']) for hoi_ann in hoi_anns]
    feat_list = split_vl(feats, node_frame_chunk_sizes)

    for hoi_ann, feat in zip(hoi_anns, feat_list):
        node_attr = feat
        num_nodes = len(node_attr)
        edge_idx = build_edge_idx(num_nodes)
        # data = Data(edge_index=hoi_ann.orc_edge_index, x=node_attr,
        #             orc_node_attr=hoi_ann.orc_node_attr, orc_edge_attr=hoi_ann.orc_edge_attr)
        if oracle:
            data = Data(edge_index=edge_idx, x=node_attr, orc_node_attr=hoi_ann.orc_node_attr)
        else:
            data = Data(edge_index=edge_idx, x=node_attr)
        data_list.append(data)
    data = Batch.from_data_list(data_list=data_list)
    batch_hoi = data.batch

    delattr(data, 'batch')
    setattr(data, 'sact_cids', torch.tensor(int(sact_cid)))
    setattr(data, 'sact_id', torch.tensor(int(id)))
    setattr(data, 'batch_hoi', batch_hoi)

    return data

def to_pyg_data_act(act_cid, id, sact_ids, feats, moma_api, oracle=True):
    data_list = []

    for i in range(len(sact_ids)):
        sact_id = sact_ids[i]
        hoi_ids = moma_api.get_ids_hoi(ids_sact=[sact_id])
        if oracle:
            hoi_anns = moma_api.get_anns_hoi(hoi_ids)
        else:
            hoi_anns = {}
            for hoi_id in hoi_ids:
                hoi_anns["actors"] = torch.load(os.path.join(PRED_ACT_DIR, hoi_id))
                hoi_anns["objects"] = torch.load(os.path.join(PRED_OBJ_DIR, hoi_id))
        graph = to_pyg_data(hoi_anns, 0, 0, feats[i], oracle)
        data_list.append(graph)
    data = Batch.from_data_list(data_list=data_list)

    delattr(data, 'batch')
    setattr(data, 'act_cid', torch.tensor(int(act_cid)))
    setattr(data, 'act_id',  id)

    return data

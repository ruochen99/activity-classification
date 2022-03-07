import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

import utils

class MLP(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(MLP, self).__init__()
    self.linear = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
    self.bn = nn.BatchNorm1d(out_dim)

  def forward(self, x):
    x = self.linear(x)
    x = self.bn(x)
    return x
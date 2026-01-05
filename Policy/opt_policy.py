from pathlib import Path
import sys
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from model_pickle.models import Models
import torch
from torch import nn
from torch_geometric.nn import GCNConv

class opt_policy(torch.nn.Module):#추가해야함
    def __init__(self,in_channels,hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head  = torch.nn.Linear(hidden, 1)
        self.cash=torch.nn.Parameter(torch.zeros(1))
        self.Models=Models()
    def forward(self, data):
        nodes=self.Models.concat(data)
        x,edge_index=nodes,nodes
        h = nn.relu(self.conv1(x, edge_index))
        h = nn.relu(self.conv2(h, edge_index))
        score=self.head(h).squeeze(-1)
        scores=torch.cat([self.cash,score],dim=0)
        weights=nn.Softmax(scores,dim=0)
        return weights

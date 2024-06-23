import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch.nn import BatchNorm1d, Linear, SmoothL1Loss

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch.nn import BatchNorm1d, Linear
import torch
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, GATConv
from torch.nn import BatchNorm1d, Linear
import torch.nn.init as init  # Add this import

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch.nn import BatchNorm1d, Linear

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch_geometric.data import Data
# Parameters
min_assets = 4
max_assets = 4
class GNN(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, 16, heads=4, concat=True, dropout=0.1)
        self.bn1 = BatchNorm1d(16 * 4)
        self.conv2 = GATConv(16 * 4, 32, heads=4, concat=True, dropout=0.1)
        self.bn2 = BatchNorm1d(32 * 4)
        
        self.attention_pooling = AttentionalAggregation(gate_nn=Linear(32 * 4, 1))
        self.fc1 = Linear(32 * 4, 32)
        self.dropout = torch.nn.Dropout(p=0.1)
        
        self.value_stream = Linear(32, 1)
        self.advantage_stream = Linear(32, num_actions)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        
        x = self.attention_pooling(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn import GATConv, AttentionalAggregation
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch.nn import BatchNorm1d, Linear

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention
from torch.nn import BatchNorm1d, Linear
import torch
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention, GATConv
from torch.nn import BatchNorm1d, Linear
import torch.nn.init as init  # Add this import

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=8, concat=True, dropout=0.1)
        self.bn1 = BatchNorm1d(32 * 8)
        self.conv2 = GATConv(32 * 8, 64, heads=8, concat=True, dropout=0.1)
        self.bn2 = BatchNorm1d(64 * 8)
        self.conv3 = GATConv(64 * 8, 32, heads=8, concat=True, dropout=0.1)
        self.bn3 = BatchNorm1d(32 * 8)

        self.attention_pooling = GlobalAttention(gate_nn=Linear(32 * 8, 1))
        self.fc1 = Linear(32 * 8, 16)
        self.fc2 = Linear(16, num_actions)
        self.dropout = torch.nn.Dropout(p=0.1)

    #     self._initialize_weights()

    # def _initialize_weights(self):
    #     init.kaiming_uniform_(self.fc1.weight, a=0.1)
    #     init.kaiming_uniform_(self.fc2.weight, a=0.1)
    #     if self.fc1.bias is not None:
    #         self.fc1.bias.data.fill_(0.01)
    #     if self.fc2.bias is not None:
    #         self.fc2.bias.data.fill_(0.01)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)

        x = self.attention_pooling(x, data.batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

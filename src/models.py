import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import GCNConv, GATConv

# =============================
# MLP (node features only)
# =============================
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    # Return node embeddings (before classifier) — used by LM head if ever needed
    # Keep signature compatible with GNN backbones.
    def encode(self, x, edge_index: Optional[torch.Tensor] = None):
        h = self.lin1(x).relu()
        h = self.lin2(h).relu()
        return h  # (N, hidden_dim) or (B, Nodes, hidden_dim)

    # Original behavior for your notebooks (node classification/logits)
    def forward(self, x):
        h = self.encode(x)
        output = self.out_layer(h).squeeze()
        return output


# =============================
# GCN
# =============================
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    # Node embeddings after graph convs
    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return h  # (N, hidden_dim)

    # Original behavior (node logits)
    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        out = self.classifier(h).squeeze()
        return out


# =============================
# GCN with learnable Edge Weights (GCN-EW)
# =============================
class GCN_EW(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index):
        super().__init__()
        torch.manual_seed(1234)
        # one learnable weight per edge in the static training graph
        self.edge_weight = nn.Parameter(torch.zeros(edge_index.shape[1]))
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def encode(self, x, edge_index):
        w = torch.exp(self.edge_weight)  # ensure positivity
        h = self.conv1(x, edge_index, w).relu()
        h = self.conv2(h, edge_index, w).relu()
        return h

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        out = self.classifier(h).squeeze()
        return out


# =============================
# GAT
# =============================
class GAT(nn.Module):
    def __init__(self, hidden_channels, heads, in_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GATConv(in_dim, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.classifier = nn.Linear(heads * hidden_channels, out_dim)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        return h

    def forward(self, x, edge_index):
        h = self.encode(x, edge_index)
        out = self.classifier(h).squeeze()
        return out


# =============================
# Lateral Movement Edge Head
# (binary edge classification on (src, dst) pair)
# =============================
class LMEdgeHead(nn.Module):
    """
    Inputs:
        z_src: (E, D) embeddings of source nodes
        z_dst: (E, D) embeddings of destination nodes
        e_attr: (E, Fe) optional edge features (e.g., protocol/flags/time features)

    Output:
        logits: (E,) raw scores; apply sigmoid for probabilities
    """
    def __init__(self, node_embed_dim: int, edge_feat_dim: int = 0, hidden: int = 128):
        super().__init__()
        in_dim = node_embed_dim * 2 + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_src, z_dst, e_attr: Optional[torch.Tensor] = None):
        if e_attr is not None:
            x = torch.cat([z_src, z_dst, e_attr], dim=-1)
        else:
            x = torch.cat([z_src, z_dst], dim=-1)
        return self.mlp(x).squeeze(-1)  # (E,)

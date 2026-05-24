import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool

class EGNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='mean') # Mean aggregation stabilizes gradients in dense graphs
        
        # Edge MLP learns from interacting node features and their invariant distance
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Node MLP updates the node representation based on its surroundings
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, pos, edge_index):
        row, col = edge_index
        # Calculate SE(3) Invariant squared distance between connected C-alphas
        radial = torch.sum((pos[row] - pos[col])**2, dim=1, keepdim=True)
        return self.propagate(edge_index, h=h, radial=radial)

    def message(self, h_i, h_j, radial):
        # h_i: receiving node, h_j: sending node, radial: distance
        feat = torch.cat([h_i, h_j, radial], dim=-1)
        return self.edge_mlp(feat)

    def update(self, aggr_out, h):
        return self.node_mlp(torch.cat([h, aggr_out], dim=-1))


class FreeEnergyEstimator(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        # Map the pre-calculated sequence embedding to our working dimension
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(n_layers)])
        
        # Thermodynamic Readout (Predicts the scalar G)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, batch):
        h = self.proj_in(batch.x)
        pos = batch.pos
        edge_index = batch.edge_index

        for layer in self.layers:
            h = layer(h, pos, edge_index)

        # Pool the atomic representations into a single global protein representation
        h_global = global_mean_pool(h, batch.batch)
        
        # Predict the relative free energy
        pred_g = self.readout(h_global).squeeze(-1)
        return pred_g
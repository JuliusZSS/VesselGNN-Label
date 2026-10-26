"""GNN classifiers with node-, segment- (region-) and centerline-existence heads."""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


class _MultiHeadGNN(torch.nn.Module):
    def __init__(self, conv, num_node_features, hidden_dim, num_classes, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList([conv(num_node_features, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(conv(hidden_dim, hidden_dim))
        self.node_fc = torch.nn.Linear(hidden_dim, num_classes)
        self.region_fc = torch.nn.Linear(hidden_dim, num_classes)
        self.centerline_fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, region_batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        node_out = F.log_softmax(self.node_fc(x), dim=1)
        region_out = F.log_softmax(self.region_fc(global_mean_pool(x, region_batch)), dim=1)
        centerline_out = torch.sigmoid(self.centerline_fc(global_mean_pool(x, batch)))
        return node_out, region_out, centerline_out


class GCN(_MultiHeadGNN):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=2):
        super().__init__(GCNConv, num_node_features, hidden_dim, num_classes, num_layers)


class GraphSAGE(_MultiHeadGNN):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=4):
        super().__init__(SAGEConv, num_node_features, hidden_dim, num_classes, num_layers)

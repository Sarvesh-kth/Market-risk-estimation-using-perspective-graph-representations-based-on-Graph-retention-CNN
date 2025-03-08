import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch as dglnn


device = 'cpu'
class TemporalEdgeConv(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(TemporalEdgeConv, self).__init__()
        self.linear1 = nn.Linear(in_features, 512)
        self.linear2 = nn.Linear(512 + out_features * 2, out_features)
        self.attn = dglnn.GATConv(out_features, out_features, num_heads=num_heads)

    def forward(self, src, dst, edge_attr):
        edge_attr = self.linear1(edge_attr).to(device)
        x = torch.cat([src, dst, edge_attr], dim=-1)
        x = self.linear2(x)
        return x

class RiskPredictionModel(nn.Module):
    def __init__(self, in_features_dict, hidden_features, num_classes, num_heads=4):
        super(RiskPredictionModel, self).__init__()
        self.projections1 = nn.ModuleDict({
            ntype: nn.Sequential(nn.ELU(), nn.Linear(in_features, hidden_features), nn.ReLU())
            for ntype, in_features in in_features_dict.items()
        })
        self.projections2 = nn.ModuleDict({
            ntype: nn.Sequential(nn.ELU(), nn.Linear(8, hidden_features), nn.ReLU())
            for ntype, in_features in in_features_dict.items()
        })
        self.temporal_edge_convs = {
            'day': TemporalEdgeConv(4, hidden_features, num_heads),
            'month': TemporalEdgeConv(6, hidden_features, num_heads),
            'quarter': TemporalEdgeConv(2, hidden_features, num_heads),
            'year': TemporalEdgeConv(3, hidden_features, num_heads)
        }

        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(hidden_features, hidden_features // num_heads, num_heads=num_heads, residual=True)
            for etype in [('day', 'to', 'month'), ('month', 'to', 'quarter'), ('quarter', 'to', 'year'),('year', 'to', 'company')]
        }, aggregate='mean').to(device)
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GATConv(hidden_features, hidden_features // num_heads, num_heads=num_heads, residual=True)
            for etype in [('day', 'to', 'month'), ('month', 'to', 'quarter'), ('quarter', 'to', 'year'),('year', 'to', 'company')]
        }, aggregate='mean').to(device)
        self.elu = nn.ELU()
        self.linear = nn.Linear(128, num_classes)

    def forward(self, graph):
        graph = graph.to(device)

        h_dict = {ntype: self.projections1[ntype](graph.x_dict[ntype]) for ntype in graph.x_dict}
        edge_index_dict = {
            ('year', 'to', 'company'): (torch.tensor([0, 1], dtype=torch.long), torch.tensor([0, 0], dtype=torch.long, device=device)),
            ('quarter', 'to', 'year'): (
                torch.tensor(graph.edge_index_dict[('quarter', 'to', 'year')][0], dtype=torch.long, device=device),
                torch.tensor(graph.edge_index_dict[('quarter', 'to', 'year')][1], dtype=torch.long, device=device)),
            ('month', 'to', 'quarter'): (
                torch.tensor(graph.edge_index_dict[('month', 'to', 'quarter')][0], dtype=torch.long, device=device),
                torch.tensor(graph.edge_index_dict[('month', 'to', 'quarter')][1], dtype=torch.long, device=device)),
            ('day', 'to', 'month'): (torch.tensor(graph.edge_index_dict[('day', 'to', 'month')][0], dtype=torch.long, device=device),
                                     torch.tensor(graph.edge_index_dict[('day', 'to', 'month')][1], dtype=torch.long, device=device))
        }
        dgl_graph = dgl.heterograph(edge_index_dict)

        for etype in dgl_graph.etypes:
            if etype in self.temporal_edge_convs:
                src, dst = dgl_graph.edges(etype=etype)
                edge_attr = graph[etype].edge_attr.view(-1, graph[etype].edge_attr.shape[-1])
                edge_attr = edge_attr.float()
                edge_feat = self.temporal_edge_convs[etype[1]](h_dict[etype[0]][src], h_dict[etype[2]][dst], edge_attr)
                h_dict[etype[2]][dst] += edge_feat

        h_dict = self.conv1(dgl_graph, h_dict)
        for ntype in h_dict:
            h_dict[ntype] = self.elu(self.projections2[ntype](h_dict[ntype]))
        h_dict = self.conv2(dgl_graph, h_dict)
        for ntype in h_dict:
            dgl_graph.nodes[ntype].data['h'] = h_dict[ntype]

        aggregated_features = dgl.mean_nodes(dgl_graph, 'h', ntype='company')
        aggregated_features = aggregated_features.view(1,
                                                       -1)
        out = self.linear(aggregated_features)
        return torch.log_softmax(out, dim=-1)

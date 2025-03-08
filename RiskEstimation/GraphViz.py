from datetime import date, timedelta
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt

data = torch.load('modgraph/modified4/A_modified.pt')
ticker = 'A'

G = nx.DiGraph()

# Process edges and nodes from the graph
for node_type in ['day', 'month', 'quarter', 'year']:
    num_nodes = data[node_type].num_nodes
    for node_idx in range(num_nodes):
        if node_type == 'day':
            date_f = date(2023, 10, 9) - timedelta(num_nodes - 1 - node_idx)
            node_id = f"Day {date_f}"
        elif node_type == 'month':
            node_id = f"Month {1 + node_idx % 12}_{2022 + node_idx // 12}"
        elif node_type == 'quarter':
            node_id = f"{2022 + node_idx // 4}_Q{1 + node_idx % 4}"
        elif node_type == 'year':
            node_id = f"Year {2022 + node_idx}"

        G.add_node(node_id, type=node_type)

mde = 'month-day'
qme = 'quarter-month'
yqe = 'year-quarter'
cye = 'company-year'

for edge_type, edge_indices in data.edge_index_dict.items():
    src_type, _, dest_type = edge_type
    for src_index, dest_index in edge_indices.T:
        src_index, dest_index = src_index.item(), dest_index.item()

        if src_type == 'day' and dest_type == 'month':
            src_id = f"Day {date(2023, 10, 9) - timedelta(546 - src_index)}"
            dest_id = f"Month {1 + dest_index % 12}_{2022 + dest_index // 12}"
            G.add_edge(src_id, dest_id, type=mde)

        elif src_type == 'month' and dest_type == 'quarter':
            src_id = f"Month {1 + src_index % 12}_{2022 + src_index // 12}"
            dest_id = f"{2022 + dest_index // 4}_Q{1 + dest_index % 4}"
            G.add_edge(src_id, dest_id, type=qme)

        elif src_type == 'quarter' and dest_type == 'year':
            src_id = f"{2022 + src_index // 4}_Q{1 + src_index % 4}"
            dest_id = f"Year {2022 + dest_index}"
            G.add_edge(src_id, dest_id, type=yqe)

        elif src_type == 'year' and dest_type == 'company':
            src_id = f"Year {2022 + src_index}"
            dest_id = f"Company_{ticker}"
            G.add_edge(src_id, dest_id, type=cye)

if not G.nodes:
    print("No nodes were added.")
if not G.edges:
    print("No edges were added.")

if G.nodes and G.edges:
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, font_color="black",
            font_weight="bold", edge_color="gray")
    plt.title("HeteroData Graph Visualization")
    plt.show()
else:
    print("Graph is empty, nothing to display.")

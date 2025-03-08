import pandas as pd
import torch
from torch_geometric.data import HeteroData
from datetime import date

iG = torch.load('modgraph/modified4/A_modified.pt')
date_to_find = date(2023, 8, 14)
sdate = date(2023,10,9)
print(iG)



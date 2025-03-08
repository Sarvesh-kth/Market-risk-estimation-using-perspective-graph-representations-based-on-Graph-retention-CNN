import pandas as pd
import torch
from torch_geometric.data import HeteroData
from datetime import date

iG = torch.load('Graphs/A_oG.pt')

def find_and_print_features(graph, Date,sdate):

    numday = (sdate - Date)
    day_idx = 182 - numday.days
    print('Number of days', day_idx)
    day_features = graph['day'].x[day_idx]
    month_idx = graph['day', 'month'].edge_index[1][graph['day', 'month'].edge_index[0] == day_idx].item()
    month_features = graph['month'].x[month_idx]
    quarter_idx = graph['month', 'quarter'].edge_index[1][graph['month', 'quarter'].edge_index[0] == month_idx].item()
    quarter_features = graph['quarter'].x[quarter_idx]

    print("Day Features:")
    print(day_features)
    print("Month Features:")
    print(month_features)
    print("Quarter Features:")
    print(quarter_features)


date_to_find = date(2023, 11, 1)  
sdate = date(2024,4,9)
find_and_print_features(iG, date_to_find,sdate)

import pandas as pd
import torch
from torch_geometric.data import HeteroData

data_paths = {
    'balance_annual': 'DATA/Filtered/balance_annual.csv',
    'cashflow_quarter': 'DATA/Filtered/cashflow_quarter.csv',
    'derived_quarter': 'DATA/Filtered/derived_quarter.csv',
    'income_quarter': 'DATA/Filtered/income_quarter.csv',
    'revDsp_daily': 'DATA/Filtered/revDsp_daily.csv',
    'revsp_daily': 'DATA/Filtered/revsp_daily.csv',
    'news_articles': 'DATA/Articles/A.csv'
}

def load_data(path, ticker='A', date_col='Date'):
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
        df.set_index(date_col, inplace=True)
    elif 'Fiscal Year' in df.columns and 'Fiscal Period' in df.columns:
        df['Fiscal Year'] = pd.to_datetime(df['Fiscal Year'].astype(str), format='%Y')
        df['Fiscal Period'] = df['Fiscal Period'].apply(lambda x: f"Q{x}")
        df.set_index(['Fiscal Year', 'Fiscal Period'], inplace=True)
    if 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]
    return df

financial_data = {key: load_data(path) for key, path in data_paths.items()}

def create_heterogeneous_graph(financial_data, start_date, end_date):
    graph = HeteroData()

    graph['day', 'day'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['day', 'month'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['month', 'quarter'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['quarter', 'year'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['year', 'company'].edge_index = torch.empty((2, 0), dtype=torch.long)

    date_range = pd.date_range(start=start_date, end=end_date)

    graph['company'].x = torch.tensor([[1]])
    graph['year'].x = torch.empty((0, 4))
    graph['quarter'].x = torch.empty((0, 11))
    graph['month'].x = torch.empty((0, 2))
    graph['day'].x = torch.empty((0, 4))


    month_indices = {}
    quarter_indices = {}
    year_indices = {}
    year_counter = 0
    year_idx = None
    prev_day_idx = -1
    for date in date_range:

        daily_data = financial_data['revsp_daily'].loc[date, ['Close', 'Volume']] if date in financial_data['revsp_daily'].index else [0, 0]
        derived_data = financial_data['revDsp_daily'].loc[date, ['Price to Earnings Ratio (ttm)', 'EV/EBITDA']] if date in financial_data['revDsp_daily'].index else [0, 0]

        daily_features = torch.tensor(daily_data.tolist() + derived_data.tolist(), dtype=torch.float)


        day_idx = len(graph['day'].x)
        graph['day'].x = torch.cat([graph['day'].x, daily_features.unsqueeze(0)], dim=0)

        if prev_day_idx != -1:
            edge = torch.tensor([[prev_day_idx, day_idx]], dtype=torch.long)
            graph['day', 'day'].edge_index = torch.cat([graph['day', 'day'].edge_index, edge.view(-1,1)], dim=1)
        prev_day_idx = day_idx


        month_label = date.strftime('%Y-%m')
        quarter_label = f"{date.year}-Q{((date.month - 1) // 3 + 1)}"
        year_label = str(date.year)


        if month_label not in month_indices:
            month_idx = len(graph['month'].x)
            graph['month'].x = torch.cat([graph['month'].x, torch.zeros((1, 2), dtype=torch.float)], dim=0)
            month_indices[month_label] = month_idx
            if graph['day', 'month'].edge_index.size(1) == 0:  
                graph['day', 'month'].edge_index = torch.tensor([[day_idx, month_idx]],
                                                                dtype=torch.long).T  
            else:
                graph['day', 'month'].edge_index = torch.cat(
                    [graph['day', 'month'].edge_index, torch.tensor([[day_idx, month_idx]], dtype=torch.long).view(-1,1)], dim=1)


        if quarter_label not in quarter_indices:
            quarter_features = get_quarter_features(financial_data, quarter_label)
            quarter_idx = len(graph['quarter'].x)
            graph['quarter'].x = torch.cat([graph['quarter'].x, quarter_features], dim=0)
            quarter_indices[quarter_label] = quarter_idx
            if graph['month', 'quarter'].edge_index.size(1) == 0:
                graph['month', 'quarter'].edge_index = torch.tensor([[month_idx, quarter_idx]], dtype=torch.long).T
            else:
                graph['month', 'quarter'].edge_index = torch.cat(
                    [graph['month', 'quarter'].edge_index, torch.tensor([[month_idx, quarter_idx]], dtype=torch.long).view(-1,1)],
                    dim=1)

        if year_label not in year_indices:
            year_features = get_year_features(financial_data, year_label)
            graph['year'].x = torch.cat([graph['year'].x, year_features], dim=0)
            year_indices[year_label] = year_counter
            year_counter += 1
            year_idx = year_counter - 1
            if graph['quarter', 'year'].edge_index.size(1) == 0:
                graph['quarter', 'year'].edge_index = torch.tensor([[quarter_idx, year_idx]], dtype=torch.long).T
            else:
                graph['quarter', 'year'].edge_index = torch.cat(
                    [graph['quarter', 'year'].edge_index, torch.tensor([[quarter_idx, year_idx]], dtype=torch.long).view(-1,1)],
                    dim=1)

    for year_idx in year_indices.values():
        current_num_edges = graph[('year', 'company')].edge_index.size(1)  
        if current_num_edges == 0:
            graph[('year', 'company')].edge_index = torch.tensor([[year_idx, 0]], dtype=torch.long).T
        else:
            x = torch.tensor([[year_idx, current_num_edges]], dtype=torch.long)
            x = x.reshape(-1,1)
            graph[('year', 'company')].edge_index = torch.cat(
                [graph[('year', 'company')].edge_index,x], dim=1)
    return graph


def get_quarter_features(financial_data, quarter_label):
    income_data = financial_data['income_quarter'].loc[quarter_label] if quarter_label in financial_data['income_quarter'].index else {
        'Operating Income (Loss)': 0, 'Net Income': 0, 'Interest Expense, Net': 0}
    cashflow_data = financial_data['cashflow_quarter'].loc[quarter_label] if quarter_label in financial_data['cashflow_quarter'].index else {
        'Net Cash from Operating Activities': 0, 'Net Cash from Financing Activities': 0, 'Change in Working Capital': 0}
    derived_data = financial_data['derived_quarter'].loc[quarter_label] if quarter_label in financial_data['derived_quarter'].index else {
        'EBITDA': 0, 'Net Profit Margin': 0, 'Return on Equity': 0, 'Debt Ratio': 0, 'Net Debt / EBITDA': 0}

    quarter_features = torch.tensor([
        income_data['Operating Income (Loss)'], income_data['Net Income'], income_data['Interest Expense, Net'],
        cashflow_data['Net Cash from Operating Activities'], cashflow_data['Net Cash from Financing Activities'], cashflow_data['Change in Working Capital'],
        derived_data['EBITDA'], derived_data['Net Profit Margin'], derived_data['Return on Equity'], derived_data['Debt Ratio'], derived_data['Net Debt / EBITDA']
    ], dtype=torch.float)
    quarter_features = quarter_features.view(-1,11)

    return quarter_features

def get_year_features(financial_data, year_key):
    annual_data = financial_data['balance_annual'].loc[year_key] if year_key in financial_data[
        'balance_annual'].index else pd.Series({
        'Total Assets': 0, 'Total Liabilities': 0, 'Total Equity': 0, 'Long Term Debt': 0
    })


    year_features = torch.tensor([
        annual_data['Total Assets'],
        annual_data['Total Liabilities'],
        annual_data['Total Equity'],
        annual_data['Long Term Debt']
    ], dtype=torch.float)
    year_features = year_features.unsqueeze(0)
    year_features = year_features.view(-1,4)

    return year_features
iG = create_heterogeneous_graph(financial_data, '2022-04-11', '2023-10-09')
gtG = create_heterogeneous_graph(financial_data, '2023-10-10', '2024-04-09')

print("Input Graph:", iG)
print("Ground Truth Graph:", gtG)

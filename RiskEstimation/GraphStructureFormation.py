import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os

data_paths = {
    'balance_annual': 'DATA/Normalized/balance_annual.csv',
    'cashflow_quarter': 'DATA/Normalized/cashflow_quarter.csv',
    'derived_quarter': 'DATA/Normalized/derived_quarter.csv',
    'income_quarter': 'DATA/Normalized/income_quarter.csv',
    'revDsp_daily': 'DATA/Normalized/revDsp_daily.csv',
    'revsp_daily': 'DATA/Normalized/revsp_daily.csv',
    'news_articles': 'DATA/Articles'
}


def get_tickers(articles_folder):
    tickers = []
    for filename in os.listdir(articles_folder):
        if filename.endswith('.csv'):
            tickers.append(filename[:-4])
    return tickers


def load_data(path, ticker='A', date_col='date'):
    head, tail = os.path.split(path)
    if tail == "Articles":
        path = os.path.join(path, f"{ticker}.csv")
        df = pd.read_csv(path).copy()
    else:
        df = pd.read_csv(path).copy()

    if 'date' in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d')
    if 'Ticker' in df.columns:
        df = df[df['Ticker'] == ticker]

    return df



def get_news_features(news_df, date_):

    date_ = date_.strftime('%Y-%m-%d')


    news_df['date'] = pd.to_datetime(news_df['date'].copy())
    news_df['date'] = news_df['date'].dt.strftime('%Y-%m-%d').copy()

    news_data = news_df[news_df['date'] == date_].copy() if date_ in news_df['date'].values else pd.DataFrame()
    news_features = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).view(-1,
                                                                                           10)

    if not news_data.empty:
        news_features[0][0] = torch.tensor(news_data['Incident Fraud'].values)
        news_features[0][1] = torch.tensor(news_data['Incident Regulatory Issues'].values)
        news_features[0][2] = torch.tensor(news_data['Incident Product Failures/Recalls'].values)
        news_features[0][3] = torch.tensor(news_data['Incident Data Breaches/Cyberattacks'].values)
        news_features[0][4] = torch.tensor(news_data['Incident Data Breaches/Cyberattacks'].values)
        news_features[0][5] = torch.tensor(news_data['Incident Management Scandals'].values)
        news_features[0][6] = torch.tensor(news_data['Incident Mergers and Acquisitions'].values)
        news_features[0][7] = torch.tensor(news_data['Incident Product Launches'].values)
        news_features[0][8] = torch.tensor(news_data['Incident Expansion into New Markets'].values)
        news_features[0][9] = torch.tensor(news_data['Incident Strategic Partnerships'].values)
    return news_features


def create_heterogeneous_graph(financial_data, ticker, ind, start_date, end_date):
    graph = HeteroData()

    graph['month', 'day'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['quarter', 'month'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['year', 'quarter'].edge_index = torch.empty((2, 0), dtype=torch.long)
    graph['company', 'year'].edge_index = torch.empty((2, 0), dtype=torch.long)

    date_range = pd.date_range(start=start_date, end=end_date)
    graph['company'].x = torch.tensor([[1.0]], dtype=torch.float)
    graph['year'].x = torch.empty((0, 4), dtype=torch.float)
    graph['quarter'].x = torch.empty((0, 11), dtype=torch.float)
    graph['month'].x = torch.empty((0, 1), dtype=torch.float)
    graph['day'].x = torch.empty((0, 14), dtype=torch.float)

    month_indices = {}
    quarter_indices = {}
    year_indices = {}
    year_counter = 0
    year_idx = 0
    prev_day_idx = -1
    for d in range(0, len(date_range)):
        date_ = date_range[d]

        month_label = date_.strftime('%Y-%m')
        quarter_label = f"{date_.year}-Q{((date_.month - 1) // 3 + 1)}"
        year_label = int(date_.year)

        if year_label not in year_indices:
            column_names = financial_data['balance_annual'].columns.to_list()

            year_features = get_year_features(financial_data, year_label, ticker)
            graph['year'].x = torch.cat([graph['year'].x, year_features], dim=0)
            # graph['year']
            year_indices[year_label] = year_counter
            year_counter += 1
            year_idx = year_counter - 1

        if quarter_label not in quarter_indices:
            quarter_features = get_quarter_features(financial_data, quarter_label, year_label, ticker)
            quarter_idx = len(graph['quarter'].x)
            graph['quarter'].x = torch.cat([graph['quarter'].x, quarter_features], dim=0)
            graph['quarter'].ql = quarter_label
            ko = 2021 + year_counter

            quarter_indices[quarter_label] = quarter_idx
            if graph['year', 'quarter'].edge_index.size(1) == 0:
                graph['year', 'quarter'].edge_index = torch.tensor([[year_idx, quarter_idx]], dtype=torch.long).T
            else:
                graph['year', 'quarter'].edge_index = torch.cat(
                    [graph['year', 'quarter'].edge_index,
                     torch.tensor([[year_idx, quarter_idx]], dtype=torch.long).view(-1, 1)],
                    dim=1)

        if month_label not in month_indices:

            year, month = month_label.split('-')
            month_value = int(month)
            m = torch.tensor(month_value, dtype=torch.long).view(1, 1)

            month_idx = len(graph['month'].x)
            graph['month'].x = torch.cat([graph['month'].x, torch.tensor(month_idx).view(1, 1)], dim=0)
            graph['month'].ml = month_label

            month_indices[month_label] = month_idx

            if graph['quarter', 'month'].edge_index.size(1) == 0:
                graph['quarter', 'month'].edge_index = torch.tensor([[quarter_idx, month_idx]], dtype=torch.long).T
            else:
                graph['quarter', 'month'].edge_index = torch.cat(
                    [graph['quarter', 'month'].edge_index,
                     torch.tensor([[quarter_idx, month_idx]], dtype=torch.long).view(-1, 1)],
                    dim=1)


        financial_data['revsp_daily']['Date'] = pd.to_datetime(financial_data['revsp_daily']['Date'], format='%Y-%m-%d')
        financial_data['revDsp_daily']['Date'] = pd.to_datetime(financial_data['revDsp_daily']['Date'],
                                                                format='%Y-%m-%d')
        daily_data = financial_data['revsp_daily'][financial_data['revsp_daily']['Date'] == date_]
        derived_data = financial_data['revDsp_daily'][financial_data['revDsp_daily']['Date'] == date_]
        daily_features = torch.tensor([
            0, 0, 0, 0
        ], dtype=torch.float).view(-1, 4)


        daily_features[0][0] = torch.tensor(daily_data['Close'].values)
        daily_features[0][1] = torch.tensor(daily_data['Volume'].values)
        daily_features[0][2] = torch.tensor(derived_data['Price to Earnings Ratio (ttm)'].values)
        daily_features[0][3] = torch.tensor(derived_data['EV/EBITDA'].values)

        news_features = get_news_features(financial_data['news_articles'], date_)

        daily_features = torch.cat((daily_features, news_features),
                                   dim=1)

        day_idx = len(graph['day'].x)
        graph['day'].x = torch.cat([graph['day'].x, daily_features], dim=0)
        graph['day'].l = date_range[d]


        if graph['month', 'day'].edge_index.size(1) == 0:
            graph['month', 'day'].edge_index = torch.tensor([[month_idx, day_idx]],
                                                            dtype=torch.long).T
        else:
            graph['month', 'day'].edge_index = torch.cat(
                [graph['month', 'day'].edge_index, torch.tensor([[month_idx, day_idx]], dtype=torch.long).view(-1, 1)],
                dim=1)

    for year_idx in year_indices.values():
        current_num_edges = graph[('company', 'year')].edge_index.size(1)
        if current_num_edges == 0:
            graph[('company', 'year')].edge_index = torch.tensor([[ind, year_idx]], dtype=torch.long).T
        else:
            x = torch.tensor([[current_num_edges, year_idx]], dtype=torch.long)
            x = x.reshape(-1, 1)
            graph[('company', 'year')].edge_index = torch.cat(
                [graph[('company', 'year')].edge_index,  torch.tensor([[ind, year_idx]], dtype=torch.long).view(-1, 1)],
                dim=1)
    return graph




def get_quarter_features(financial_data, quarter_labels, year_label, ticker):
    income_data = pd.read_csv(
        r"C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Normalized\income_quarter.csv").copy()

    cashflow_data = pd.read_csv(
        r"C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Normalized\cashflow_quarter.csv").copy()

    derived_data = pd.read_csv(
        r"C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Normalized\derived_quarter.csv").copy()
    _, quarter_label = str.split(quarter_labels, '-', 1)
    income_data = income_data[(income_data['Ticker'] == ticker) &
                              (income_data['Year'] == year_label) &
                              (income_data['Fiscal Period'] == quarter_label)]

    income_data = income_data.reset_index(drop=True)
    cashflow_data = financial_data['cashflow_quarter'][
        (financial_data['cashflow_quarter']['Year'] == year_label) &
        (financial_data['cashflow_quarter']['Fiscal Period'] == quarter_label)
        ]
    cashflow_data = cashflow_data.reset_index(drop=True)
    derived_data = financial_data['derived_quarter'][
        (financial_data['derived_quarter']['Year'] == year_label) &
        (financial_data['derived_quarter']['Fiscal Period'] == quarter_label)
        ]
    derived_data = derived_data.reset_index(drop=True)
    quarter_features = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).view(1,
                                                                                                 11)  # Change dtype to long

    for i, column in enumerate(['Net Cash from Operating Activities', 'Net Cash from Financing Activities', 'Net Cash from Investing Activities',
                                'Change in Working Capital']):
        column_data = cashflow_data[column].fillna(0).astype('float32')
        quarter_features[0][i] = torch.tensor(column_data.values)

    for i, column in enumerate(['EBITDA', 'Net Profit Margin', 'Return on Equity', 'Debt Ratio']):
        column_data = derived_data[column].fillna(0).astype('float32')
        quarter_features[0][3 + i] = torch.tensor(column_data.values)

    for i, column in enumerate(['Operating Income (Loss)', 'Net Income','Revenue']):
        column_data = income_data[column].fillna(0).astype('float32')
        quarter_features[0][8 + i] = torch.tensor(column_data.values)

    return quarter_features


def get_year_features(financial_data, year_key, ticker):
    annual_data = pd.read_csv(
        r"C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Normalized\balance_annual.csv").copy()

    column_names = annual_data.columns.to_list()
    annual_data = annual_data[(annual_data['Ticker'] == ticker) & (annual_data['Year'] == year_key)]

    year_features = torch.zeros((1, 4), dtype=torch.float32)

    year_features[0][0] = torch.tensor(annual_data['Total Assets'].values, dtype=torch.float32)
    year_features[0][1] = torch.tensor(annual_data['Total Liabilities'].values, dtype=torch.float32)
    year_features[0][2] = torch.tensor(annual_data['Total Equity'].values, dtype=torch.float32)
    year_features[0][3] = torch.tensor(annual_data['Long Term Debt'].values, dtype=torch.float32)

    return year_features


articles_folder = data_paths['news_articles']
graphs_folder = 'Comgraph'

existing_graph_tickers = [
    filename.split('_')[0] for filename in os.listdir(graphs_folder) if filename.endswith('_iG.pt')
]

tickers = get_tickers(articles_folder)

tickers_to_process = [ticker for ticker in tickers if ticker not in existing_graph_tickers]

i = -1
for ticker in tickers_to_process:
    financial_data = {key: load_data(path, ticker=ticker) for key, path in data_paths.items()}
    try:
        i = i + 1
        iG = create_heterogeneous_graph(financial_data, ticker, i ,'2022-04-11', '2023-10-09')
        print("Graph Saved for Ticker ", ticker)
        torch.save(iG, f"Comgraph/{ticker}_iG.pt")
    except:
        print(ticker)
        i = i - 1
        pass

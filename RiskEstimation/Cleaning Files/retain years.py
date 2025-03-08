import pandas as pd
import os

csv_file_path = 'DATA/Normalized/revDsp_daily.csv'
graphs_directory = 'chkg'

df_filtered = pd.read_csv(csv_file_path)


tickers_with_nulls = df_filtered[df_filtered[['Price to Earnings Ratio (ttm)','EV/EBITDA']].isnull().any(axis=1)]['Ticker'].unique().tolist()

graph_files = os.listdir(graphs_directory)

for file in graph_files:
    ticker_name = file.split('_')[0]
    if ticker_name in tickers_with_nulls:
        os.remove(os.path.join(graphs_directory, file))
        print(f"Deleted graph file for ticker: {ticker_name}")

print("Completed file deletion for tickers with null entries.")

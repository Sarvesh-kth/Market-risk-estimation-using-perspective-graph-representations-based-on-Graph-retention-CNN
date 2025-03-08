import pandas as pd
import os

ground_truth_csv_path = 'clustered_data.csv'
graphs_directory = 'chkg'

ground_truth_df = pd.read_csv(ground_truth_csv_path)

graph_files = os.listdir(graphs_directory)
graph_tickers = {file.split('_')[0] for file in graph_files}

ground_truth_tickers = set(ground_truth_df['Ticker'].unique())


print(graph_tickers)
print(ground_truth_tickers)

common_tickers = graph_tickers.intersection(ground_truth_tickers)

print(ground_truth_tickers - common_tickers)

ground_truth_df = ground_truth_df[ground_truth_df['Ticker'].isin(common_tickers)]
ground_truth_df.reset_index(drop=True, inplace=True)

ground_truth_df.to_csv(ground_truth_csv_path, index=False)

for file in graph_files:
    ticker_name = file.split('_')[0]
    if ticker_name not in common_tickers:
        os.remove(os.path.join(graphs_directory, file))
        print(f"Deleted graph file for ticker: {ticker_name}")

print("Sync complete: Ground truth and graph files are now synchronized.")

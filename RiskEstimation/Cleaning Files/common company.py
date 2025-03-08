import pandas as pd
import os

directory_path = 'DATA/Normalized'

file_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]


dataframes = [pd.read_csv(fp) for fp in file_paths]

common_tickers = set(dataframes[0]['Ticker'])
for df in dataframes[1:]:
    common_tickers.intersection_update(set(df['Ticker']))

filtered_dataframes = [df[df['Ticker'].isin(common_tickers)] for df in dataframes]

for fp, df in zip(file_paths, filtered_dataframes):
    df.to_csv(fp, index=False)

print(f"All files have been updated with only common tickers. Number of common tickers: {len(common_tickers)}")

import pandas as pd
import os

revsp_df = pd.read_excel('clustered_data.xlsx')

tickers = set(revsp_df['Ticker'].tolist())
print(tickers)

articles_directory = 'DATA/Articles'

for filename in os.listdir(articles_directory):
    if filename.endswith('.csv'):
        ticker = filename.split('.')[0]

        if ticker not in tickers:
            os.remove(os.path.join(articles_directory, filename))
            print(f"Deleted {filename} as ticker {ticker} is not present revsp.csv")

import pandas as pd
import os

csv_file_path = 'DATA/Filtered/Dsp_daily.csv'
output_file_path = 'DATA/Filtered/revDsp_daily.csv'

data = pd.read_csv(csv_file_path)

data['Date'] = pd.to_datetime(data['Date'])

tickers = data['Ticker'].unique()
all_tickers_data = []

for ticker in tickers:
    ticker_data = data[data['Ticker'] == ticker]

    ticker_data.set_index('Date', inplace=True)
    full_date_range = pd.date_range(start=ticker_data.index.min(), end=ticker_data.index.max())
    ticker_data = ticker_data.reindex(full_date_range, method='ffill')

    ticker_data.reset_index(inplace=True)
    ticker_data.rename(columns={'index': 'Date'}, inplace=True)
    ticker_data['Ticker'] = ticker


    all_tickers_data.append(ticker_data)

final_data = pd.concat(all_tickers_data, ignore_index=True)

final_data.to_csv(output_file_path, index=False)

print("Missing daily values have been filled for each ticker and saved.")

import pandas as pd

filtered_file_path = 'US DATA/filtered_us-shareprices-daily.csv'

filtered_share_prices = pd.read_csv(filtered_file_path)

filtered_share_prices['Date'] = pd.to_datetime(filtered_share_prices['Date'])

unique_tickers = filtered_share_prices['Ticker'].unique()[:10]

date_ranges = pd.DataFrame(columns=['Ticker', 'Earliest Date', 'Latest Date'])

for ticker in unique_tickers:
    ticker_data = filtered_share_prices[filtered_share_prices['Ticker'] == ticker]

    earliest_date = ticker_data['Date'].min()
    latest_date = ticker_data['Date'].max()

    new_row = pd.DataFrame({
        'Ticker': [ticker],
        'Earliest Date': [earliest_date],
        'Latest Date': [latest_date]
    })
    date_ranges = pd.concat([date_ranges, new_row], ignore_index=True)
print(date_ranges)

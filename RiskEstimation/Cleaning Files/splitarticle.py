import pandas as pd

excel_file_path = 'clustered_data.xlsx'

df = pd.read_excel(excel_file_path)

unique_tickers = df['Ticker'].nunique()

print(f"Number of unique tickers: {unique_tickers}")
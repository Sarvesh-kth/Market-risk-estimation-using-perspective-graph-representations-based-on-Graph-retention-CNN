import pandas as pd
import os

balance_sheet_dir = r"C:\Users\asus\PycharmProjects\pythonProject\FYP\US DATA"
article_dir = r"C:\Users\asus\PycharmProjects\pythonProject\FYP\US DATA\Articles"

derbalance_files = [
    "us-derived-banks-quarterly-asreported.csv",
    "us-derived-insurance-quarterly-asreported.csv",
    "us-derived-quarterly-full-asreported.csv"
]
balbalance_files = [
    "us-balance-banks-annual-full-asreported.csv",
    "us-balance-insurance-annual-full-asreported.csv",
    "us-balance-annual-full-asreported.csv"
]
incbalance_files = [
    "us-income-banks-quarterly-full--asreported.csv",
    "us-income-insurance-quarterly-full-asreported.csv",
    "us-income-quarterly-full-asreported.csv"
]
cashbalance_files = [
    "us-cashflow-banks-quarterly-full-asreported.csv",
    "us-cashflow-insurance-quarterly-full-asreported.csv",
    "us-cashflow-quarterly-full-asreported.csv"
]
spbalance_files = [
    "us-shareprices-daily.csv"
]
Dspbalance_files = [
    "us-derived-shareprices-daily.csv"
]

article_files = os.listdir(article_dir)
tickers = [filename.split('.')[0] for filename in article_files]

filtered_data = pd.DataFrame()

for file in derbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)


output_file_path = os.path.join(balance_sheet_dir, 'derived_quarter.csv')
filtered_data.to_csv(output_file_path, index=False)

filtered_data = pd.DataFrame()
for file in balbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)

output_file_path = os.path.join(balance_sheet_dir, 'balance_annual.csv')
filtered_data.to_csv(output_file_path, index=False)

print("Number of tickers ",len(tickers))
print("Number of unique companies:", filtered_data['Ticker'].nunique())
filtered_data = pd.DataFrame()
for file in incbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)

output_file_path = os.path.join(balance_sheet_dir, 'income_quarter.csv')
filtered_data.to_csv(output_file_path, index=False)

print("Number of tickers ",len(tickers))
print("Number of unique companies:", filtered_data['Ticker'].nunique())
filtered_data = pd.DataFrame()
for file in cashbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)

output_file_path = os.path.join(balance_sheet_dir, 'cash_quarter.csv')
filtered_data.to_csv(output_file_path, index=False)

print("Number of tickers ",len(tickers))
print("Number of unique companies:", filtered_data['Ticker'].nunique())
filtered_data = pd.DataFrame()
for file in spbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)

output_file_path = os.path.join(balance_sheet_dir, 'sp_daily.csv')
filtered_data.to_csv(output_file_path, index=False)

print("Number of tickers ",len(tickers))
print("Number of unique companies:", filtered_data['Ticker'].nunique())
filtered_data = pd.DataFrame()
for file in Dspbalance_files:
    file_path = os.path.join(balance_sheet_dir, file)
    balance_data = pd.read_csv(file_path)

    filtered_balance_data = balance_data[balance_data['Ticker'].isin(tickers)]

    filtered_data = pd.concat([filtered_data, filtered_balance_data], ignore_index=True)

output_file_path = os.path.join(balance_sheet_dir, 'Dsp_daily.csv')
filtered_data.to_csv(output_file_path, index=False)



print("Number of tickers ",len(tickers))
print("Number of unique companies:", filtered_data['Ticker'].nunique())
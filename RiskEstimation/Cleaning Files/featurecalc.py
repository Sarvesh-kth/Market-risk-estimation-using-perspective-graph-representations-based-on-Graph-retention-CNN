import numpy as np
import pandas as pd

data_path = "merged_data/all_data_merged_filtered.csv"

combined_data = pd.read_csv(data_path)


combined_data['Current Ratio'] = combined_data['Total Current Assets'] / combined_data['Total Current Liabilities']
combined_data['Debt-to-Equity Ratio'] = combined_data['Total Debt'] / combined_data['Total Equity']

def handle_zero_division(x, y):
  return x / (y if y != 0 else 1)  

combined_data['Interest Coverage Ratio'] = combined_data.apply(lambda row: handle_zero_division(row['Operating Income (Loss)'], row['Interest Expense, Net']), axis=1)

combined_data['Net Profit Margin'] = combined_data['Net Income'] / combined_data['Revenue']
combined_data['ROE'] = combined_data['Net Income'] / combined_data['Total Equity']

output_path = "merged_data/all_data_merged_with_ratios.csv"
combined_data.to_csv(output_path, index=False)

print(f"Financial ratios calculated and saved to {output_path}")

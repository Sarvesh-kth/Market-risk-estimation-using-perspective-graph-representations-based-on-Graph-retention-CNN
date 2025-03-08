import pandas as pd
import os

directory = 'DATA/Articles'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)

        df = pd.read_csv(file_path)

        start_idx = df.columns.get_loc('Fraud')
        end_idx = df.columns.get_loc('Successful Litigations')

        new_columns = df.columns.tolist()
        new_columns[start_idx:end_idx + 1] = ['Incident ' + col for col in df.columns[start_idx:end_idx + 1]]

        df.columns = new_columns

        df.to_csv(file_path, index=False)
        print(f"Updated {filename}")

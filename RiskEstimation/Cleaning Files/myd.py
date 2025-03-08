import pandas as pd
import os

directory = 'DATA/Articles'  

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)

        df['date'] = pd.to_datetime(df['date'])

        df['day'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        df.to_csv(filepath, index=False)

        print(f'Processed and saved: {filename}')

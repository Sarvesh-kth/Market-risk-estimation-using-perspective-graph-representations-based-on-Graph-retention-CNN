import os
import pandas as pd

def process_document(file_path, output_dir, missing_threshold=0.00):
    df = pd.read_csv(file_path)
    print(file_path)
    print(df.columns.tolist())

    numeric_cols = df.select_dtypes(include='number').columns
    non_numeric_cols = df.select_dtypes(exclude='number').columns

    year_data = pd.DataFrame()
    if 'Year' in numeric_cols:
        df['SimFinId'] = df['Year']
        year_data = df['Year']
        print(numeric_cols.drop('Year'))
        numeric_cols.drop('Year')

    df_norm = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    df[numeric_cols] = df_norm

    df = pd.concat([df[non_numeric_cols], year_data, df[numeric_cols]], axis=1)
    if 'Year' in df.columns:
        df['SimFinId'] = df['Year']

    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)

input_dir = 'DATA/Filtered'
output_dir = 'DATA/Normalized'
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    process_document(file_path, output_dir)
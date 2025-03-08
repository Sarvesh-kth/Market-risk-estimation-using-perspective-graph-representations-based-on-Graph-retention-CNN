import pandas as pd

df = pd.read_csv(r'C:\Users\asus\PycharmProjects\pythonProject\FYP\DATA\Filtered\income_quarter.csv')
column_names = df.columns.to_list()
print(column_names)
from sklearn.preprocessing import StandardScaler
import pandas as pd
from k_means_constrained import KMeansConstrained

file_name = "all_data_merged_with_ratios.csv"
df = pd.read_csv(file_name)

features = ['Current Ratio', 'Debt-to-Equity Ratio', 'Interest Coverage Ratio',
            'Net Profit Margin', 'ROE', 'Price to Earnings Ratio (quarterly)', 'Price to Book Value']

data = df.dropna(subset=features)[features]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


n_clusters = 5
size_min = len(data_scaled) // n_clusters - 10  
size_max = len(data_scaled) // n_clusters + 10

kmeans = KMeansConstrained(
    n_clusters=n_clusters,
    size_min=size_min,
    size_max=size_max,
    random_state=42
)
df['Cluster'] = kmeans.fit_predict(data_scaled)


cluster_labels_map = {
    0: "Extremely Low Risk",
    1: "Low Risk",
    2: "Medium Risk",
    3: "High Risk",
    4: "Extremely High Risk"
}
df['Risk_Level'] = df['Cluster'].apply(lambda x: cluster_labels_map[x])

one_hot = pd.get_dummies(df['Cluster'], prefix='rl')
one_hot.columns = [f'rl{i}' for i in range(n_clusters)]
df = pd.concat([df, one_hot], axis=1)

df[one_hot.columns] = df[one_hot.columns].astype(int)

output_file = "clustered_data.xlsx"
df.to_excel(output_file, index=False)

cluster_counts = df['Cluster'].value_counts().sort_index()
print("Number of companies in each cluster:")
for i in range(n_clusters):
    print(f"Cluster {i} ({cluster_labels_map[i]}): {cluster_counts[i]}")

print("Data clustered and risk labels assigned. Saved to", output_file)

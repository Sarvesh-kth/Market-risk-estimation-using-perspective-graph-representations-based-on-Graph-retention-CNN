

import os
import torch
import pandas as pd
from rkm1 import RiskPredictionModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import warnings
warnings.filterwarnings('ignore')


device = torch.device('cpu')
graphs_directory = "modgraph/modified4"
tracking_file = 'performance_metric.txt'
model_path = 'trained_risk_model_rkm30.pth'

file_path = 'clustered_data.xlsx'
data = pd.read_excel(file_path)
target_columns = ['rl0', 'rl1', 'rl2', 'rl3', 'rl4']
data['filename'] = data['Ticker'].apply(lambda x: f"{x}_modified.pt")

all_files = set(os.listdir(graphs_directory))
data = data[data['filename'].isin(all_files)]
target_labels = torch.tensor(data[target_columns].values, dtype=torch.float32)

in_features_dict = {'company': 1, 'year': 4, 'quarter': 11, 'month': 1, 'day': 14}
model = RiskPredictionModel(in_features_dict, 32, 5, num_heads=4).to(device)
optimizer = Adam(model.parameters(), lr=0.006, weight_decay=0)
scheduler = ExponentialLR(optimizer, gamma=0.999)
loss_function = CrossEntropyLoss()

filename_to_index = {filename: i for i, filename in enumerate(data['filename'])}

num_epochs = 300
pt_files = [f for f in os.listdir(graphs_directory) if f.endswith('.pt')]

with open(tracking_file, "w") as f:
    f.write("Epoch\tTrain Loss\n")

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_steps = 0

    with tqdm(total=len(pt_files), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='graph') as progress_bar:
        for filename in pt_files:
            graph_path = os.path.join(graphs_directory, filename)
            graph = torch.load(graph_path).to(device)
            index = filename_to_index.get(filename)
            if index is not None:
                target = target_labels[index].unsqueeze(0).to(device)

                outputs = model(graph)
                loss = loss_function(outputs, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1
            progress_bar.update(1)


    avg_train_loss = train_loss / max(train_steps, 1)

    with open(tracking_file, "a") as f:
        f.write(f"{epoch + 1}\t{avg_train_loss:.4f}\n")

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}")
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'riskmodel2_epoch_{epoch}.pth')
        print("Saved file epoch ",epoch)



df = pd.DataFrame({
    'epoch': list(range(1, num_epochs + 1)),
    'train_loss': [avg_train_loss] * num_epochs
})
df.to_excel('training_metrics.xlsx', index=False)
torch.save(model.state_dict(), model_path)

print("Training completed and saved!")
